import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

# --- 路径与环境设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LATEST_MODEL_PATH = MODEL_DIR / "latest_cddpm_model.pth" 

IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time

# --- 1. 参数设置 ---
DEVICE = init_gpu_environment()
BATCH_SIZE = 128
NUM_EPOCHS = 500
LR = 2e-4
TIMESTEPS = 1000
BETA_START, BETA_END = 1e-4, 0.02

# CFG 参数
NUM_CLASSES = 10 
# 我们增加一个第 11 类（索引 10）作为“空条件”
CFG_DROP_PROB = 0.1 

# --- 2. 加载数据集 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform), 
                          batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 网络模块 ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalBlock(nn.Module):
    """融合了时间步和类别标签信息的残差块"""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        # 这里的 emb_dim 包含了 time_emb 和 label_emb 的融合
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU())
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, emb):
        h = self.transform(self.conv1(x))
        # 将融合后的 embedding 注入
        emb = self.mlp(emb)[(..., ) + (None, ) * 2]
        h = h + emb
        h = self.norm2(self.conv2(h))
        return self.relu(h)

class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 256 # 统一的嵌入维度
        
        # 1. 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, emb_dim),
            nn.SiLU()
        )
        
        # 2. 标签嵌入 (10个数字 + 1个空条件 = 11)
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, emb_dim)
        
        # 3. 基础卷积
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1) # 输出: 32ch
        
        # --- Down ---
        self.down1 = ConditionalBlock(32, 64, emb_dim) # 输出: 64ch (d1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = ConditionalBlock(64, 128, emb_dim) # 输出: 128ch (d2)
        self.pool2 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bot1 = ConditionalBlock(128, 128, emb_dim) # 输出: 128ch

        # --- Up ---
        # 1. 从 7x7 还原到 14x14
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1) # 输出: 64ch
        # 拼接 d2: 64ch + 128ch = 192ch
        self.up_block1 = ConditionalBlock(64 + 128, 64, emb_dim) 
        
        # 2. 从 14x14 还原到 28x28
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 输出: 32ch
        # 拼接 d1: 32ch + 64ch = 96ch (修复点在此)
        self.up_block2 = ConditionalBlock(32 + 64, 32, emb_dim)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t, y):
        # 融合时间步和标签
        t_emb = self.time_mlp(t)
        y_emb = self.label_emb(y)
        emb = t_emb + y_emb 
        
        # --- Down ---
        x_init = self.conv0(x)          # [B, 32, 28, 28]
        d1 = self.down1(x_init, emb)    # [B, 64, 28, 28]
        p1 = self.pool1(d1)             # [B, 64, 14, 14]
        
        d2 = self.down2(p1, emb)        # [B, 128, 14, 14]
        p2 = self.pool2(d2)             # [B, 128, 7, 7]

        # --- Bottleneck ---
        bot = self.bot1(p2, emb)        # [B, 128, 7, 7]

        # --- Up ---
        u1 = self.up1(bot)              # [B, 64, 14, 14]
        u1 = torch.cat([u1, d2], dim=1) # [B, 64 + 128, 14, 14] -> 192ch
        u1 = self.up_block1(u1, emb)    # [B, 64, 14, 14]

        u2 = self.up2(u1)               # [B, 32, 28, 28]
        u2 = torch.cat([u2, d1], dim=1) # [B, 32 + 64, 28, 28] -> 96ch
        u2 = self.up_block2(u2, emb)    # [B, 32, 28, 28]
        
        return self.out(u2)

# --- 4. CDDPM 框架 ---
class CDDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        # 调度器参数注册逻辑同 DDPM...
        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1. - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1. - alphas_cumprod))

    def extract(self, a, t, x_shape):
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def forward(self, x_0, y):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        # 训练过程：随机将一部分标签替换为“空标签”以支持 CFG
        # mask 逻辑：以 CFG_DROP_PROB 的概率将 y 替换为 NUM_CLASSES(10)
        mask = torch.bernoulli(torch.full(y.shape, CFG_DROP_PROB, device=y.device)).to(torch.bool)
        y_train = torch.where(mask, torch.tensor(NUM_CLASSES, device=y.device), y)

        x_t = (self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
               self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)
        
        return F.mse_loss(noise, self.model(x_t, t, y_train))

    @torch.no_grad()
    def sample(self, labels, cfg_scale=3.0):
        """
        labels: 想要生成的数字列表，如 [1, 2, 3...]
        cfg_scale: 指导强度。w=0等同于无条件，w=3是常用推荐值
        """
        self.model.eval()
        n = len(labels)
        y_cond = torch.tensor(labels, device=DEVICE)
        y_uncond = torch.full_like(y_cond, NUM_CLASSES) # 空标签

        img = torch.randn((n, 1, 28, 28), device=DEVICE)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
            
            # Classifier-Free Guidance 核心逻辑
            # 同时计算有条件和无条件噪声
            eps_cond = self.model(img, t, y_cond)
            eps_uncond = self.model(img, t, y_uncond)
            
            # 混合噪声
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            # 降噪步逻辑...
            betas_t = self.extract(self.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, img.shape)
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(img)
                var = self.extract(self.posterior_variance, t, img.shape)
                img = model_mean + torch.sqrt(var) * noise
            else:
                img = model_mean

        return torch.clamp((img + 1) / 2, 0.0, 1.0)

# --- 5. 训练函数 ---
def train_model():
    model = ConditionalUNet().to(DEVICE)
    cddpm = CDDPM(model).to(DEVICE)
    optimizer = optim.AdamW(cddpm.parameters(), lr=LR)
    
    loss_list = []
    start_epoch = 0
    if LATEST_MODEL_PATH.exists():
        ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
        cddpm.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        loss_list = ckpt['train_losses']

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        cddpm.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = cddpm(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_loss)
        logger.info(f"Epoch {epoch+1} CDDPM Loss: {avg_loss:.4f}")

    torch.save({
        'epoch': start_epoch + NUM_EPOCHS,
        'model_state_dict': cddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list
    }, LATEST_MODEL_PATH)

def load_cddpm_model() -> CDDPM:
    """封装加载 CDDPM 模型的逻辑"""
    model = ConditionalUNet().to(DEVICE)
    cddpm = CDDPM(model).to(DEVICE)
    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        cddpm.load_state_dict(checkpoint['model_state_dict'])
        cddpm.eval()
        return cddpm
    else:
        raise FileNotFoundError("未找到模型文件，请先执行训练。")

@torch.no_grad()
def generate_conditional_mode():
    """模式: 交互式条件生成"""
    try:
        cddpm = load_cddpm_model()
    except Exception as e:
        logger.error(e)
        return

    print("\n" + "-"*30)
    print("  条件生成子菜单:")
    print("  输入 [0-9]: 生成指定的单个数字")
    print("  输入 [10] : 生成 0-9 的全数字序列")
    print("  输入 [q]  : 返回主菜单")
    print("-"*30)
    
    choice = input("请输入你的选择: ").strip().lower()
    
    if choice == 'q':
        return

    try:
        val = int(choice)
        if 0 <= val <= 9:
            target_labels = [val]
            title_prefix = f"digit_{val}"
        elif val == 10:
            target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            title_prefix = "all_digits"
        else:
            logger.warning("无效输入，请输入 0-10 之间的数字。")
            return
    except ValueError:
        logger.warning("输入非法，请输入数字。")
        return

    # 采样
    logger.info(f"正在生成 {target_labels}，请稍候...")
    imgs = cddpm.sample(target_labels, cfg_scale=4.0) # 提高 cfg_scale 可以让数字更清晰
    
    # 绘图逻辑
    num_imgs = len(target_labels)
    current_time = get_current_time()
    
    if num_imgs == 1:
        # 只生成一张图
        plt.figure(figsize=(4, 4))
        plt.imshow(imgs[0].cpu().squeeze(), cmap='gray')
        plt.title(f"Generated Digit: {target_labels[0]}")
        plt.axis('off')
    else:
        # 生成 0-9 序列
        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axes[i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
            axes[i].set_title(f"L:{i}")
            axes[i].axis('off')
    
    save_path = IMAGE_DIR / f"cddpm_{title_prefix}_{current_time}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"图片已保存至: {save_path.name}")

def main():
    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      DDPM 扩散模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Denoising)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 生成损失函数图表 (Loss Plot)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == '1':
            train_model()
        elif choice == '2':
            generate_conditional_mode()
        elif choice in ['0', 'exit']:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3 或 0")


if __name__ == '__main__':
    main()