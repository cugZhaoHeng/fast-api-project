import os
import sys
import math
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

# --- 路径与环境设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
GEO_IMAGE_DIR = DATA_DIR / 'geo_images'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LATEST_MODEL_PATH = MODEL_DIR / "latest_ddpm_geo_model.pth" 

COMBINE_IMAGES_ROW: int = 3
COMBINE_IMAGES_COL: int = 3
IMAGE_WIDTH: int = 64
IMAGE_HEIGHT: int = 64

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time
# 注意：由于地质图无标签，evaluate_images (MNIST专用) 在此建议仅保留 FID 评估
from utils.fid_pr_evaluator import compute_fid_and_pr
from utils.image_dataloader import get_vae_dataloaders

# --- 1. 参数设置 ---
DEVICE = init_gpu_environment()
BATCH_SIZE = 64        # 64x64 内存占用更高，适当减小 BatchSize
NUM_EPOCHS = 100       # 地质图更复杂，建议增加训练轮数

LR = 2e-4
WEIGHT_DECAY = 1e-4
TIMESTEPS = 1000       
BETA_START = 1e-4      
BETA_END = 0.02        

# --- 2. 加载 MNIST 数据集 ---
# 设置 transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset, train_loader, test_loader = get_vae_dataloaders(GEO_IMAGE_DIR, transform)
logger.info(f"train_loader: {len(train_loader)}")
logger.info(f"test_loader: {len(test_loader)}")

# --- 3. DDPM 网络核心模块: 64x64 适配型 U-Net ---

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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        h = self.transform(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.norm2(self.conv2(h))
        return self.relu(h)

class UNet(nn.Module):
    """
    适配 64x64 的 U-Net
    下采样路径: 64x64(d1) -> 32x32(d2) -> 16x16(d3) -> 8x8(bottleneck)
    上采样路径: 8x8 -> 16x16(u1+d3) -> 32x32(u2+d2) -> 64x64(u3+d1)
    """
    def __init__(self):
        super().__init__()
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # --- Encoder (Down) ---
        self.conv0 = nn.Conv2d(1, 64, 3, padding=1)     # 64x64
        self.down1 = Block(64, 64, time_emb_dim)        # 64x64
        self.pool1 = nn.MaxPool2d(2)                    # 32x32
        
        self.down2 = Block(64, 128, time_emb_dim)       # 32x32
        self.pool2 = nn.MaxPool2d(2)                    # 16x16
        
        self.down3 = Block(128, 256, time_emb_dim)      # 16x16
        self.pool3 = nn.MaxPool2d(2)                    # 8x8

        # --- Bottleneck ---
        self.bot1 = Block(256, 256, time_emb_dim)       # 8x8

        # --- Decoder (Up) ---
        # 1. 8x8 -> 16x16
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  
        self.up_block1 = Block(128 + 256, 128, time_emb_dim) # u1(128) + d3(256)
        
        # 2. 16x16 -> 32x32
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   
        self.up_block2 = Block(64 + 128, 64, time_emb_dim)   # u2(64) + d2(128)
        
        # 3. 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    
        self.up_block3 = Block(32 + 64, 32, time_emb_dim)    # u3(32) + d1(64)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        # Encoder
        x_init = self.conv0(x)          # 64x64, 64ch
        d1 = self.down1(x_init, t)      # 64x64, 64ch
        p1 = self.pool1(d1)             # 32x32, 64ch
        
        d2 = self.down2(p1, t)          # 32x32, 128ch
        p2 = self.pool2(d2)             # 16x16, 128ch
        
        d3 = self.down3(p2, t)          # 16x16, 256ch
        p3 = self.pool3(d3)             # 8x8, 256ch

        # Bottleneck
        bot = self.bot1(p3, t)          # 8x8, 256ch

        # Decoder
        # 第一层上采样: 8x8 -> 16x16，对应拼接 d3 (16x16)
        u1 = self.up1(bot)              
        u1 = torch.cat([u1, d3], dim=1) # 128 + 256 = 384ch
        u1 = self.up_block1(u1, t)

        # 第二层上采样: 16x16 -> 32x32，对应拼接 d2 (32x32)
        u2 = self.up2(u1)               
        u2 = torch.cat([u2, d2], dim=1) # 64 + 128 = 192ch
        u2 = self.up_block2(u2, t)
        
        # 第三层上采样: 32x32 -> 64x64，对应拼接 d1 (64x64)
        u3 = self.up3(u2)               
        u3 = torch.cat([u3, d1], dim=1) # 32 + 64 = 96ch
        u3 = self.up_block3(u3, t)
        
        return self.out(u3)

# --- 4. 扩散模型调度器 (DDPM 类逻辑保持不变，只需确保 sample 形状正确) ---
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x_0):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + \
              self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, t_index):
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x_t.shape)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * self.model(x_t, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0: return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample(self, num_images):
        self.model.eval()
        img = torch.randn((num_images, 1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(img, t, i)
        img = (img + 1) / 2
        return torch.clamp(img, 0.0, 1.0)

# --- 5. 训练与业务逻辑 (基本保持原样，修正采样大小) ---
def train_model():
    unet = UNet().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    optimizer = optim.AdamW(ddpm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    loss_list = []
    start_epoch = 0
    train_times = 0
    total_training_time = 0.0
    start_time = time.time()

    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        start_epoch = checkpoint["epoch"]
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)
        logger.info(f"已加载模型, 起始 epoch={start_epoch}")

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        ddpm.train()
        avg_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = ddpm(x)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
        
        loss_list.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}] MSE Loss: {avg_loss:.6f}")

    # 保存逻辑... (同原代码)
    torch.save({
        'epoch': start_epoch + NUM_EPOCHS,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1,
        'total_training_time': total_training_time + (time.time() - start_time),
    }, LATEST_MODEL_PATH)

def load_model() -> DDPM:
    unet = UNet().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    return ddpm


def save_large_image(img_data, path: Path, title: str = None, is_compare: bool = False):
    figsize = (8, 4) if is_compare else (4, 4)
    plt.figure(figsize=figsize, dpi=100)
    plt.imshow(img_data, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def generate_mode():
    """模式 2-1: 从纯噪声生成 9 张 64x64 图片并保存宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型，请先执行训练。")
        return

    logger.info(f"开始 DDPM 采样生成 64x64 地质图，共 {TIMESTEPS} 步...")
    ddpm = load_model()
    current_timestamp = get_current_time()
    
    num_samples = COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL
    # ddpm.sample 内部已经处理了从 [-1, 1] 到 [0, 1] 的转换
    gen_imgs = ddpm.sample(num_samples).cpu().numpy()

    # 1. 保存 9 张独立大图
    for i in range(num_samples):
        filename = IMAGE_DIR / f"geo_ddpm_gen_{current_timestamp}_{i + 1:02d}.png"
        # 移除通道维度并保存
        save_large_image(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), filename)

    # 2. 保存 3*3 宫格图
    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
        ax.axis('off')
        
    grid_fn = IMAGE_DIR / f"geo_ddpm_gen_{current_timestamp}_grid.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info(f"Generate 模式运行完毕，图片已保存至 {IMAGE_DIR}")


@torch.no_grad()
def denoise_process_mode():
    """模式 2-2: 展示地质图从全噪声到清晰结构的去噪渐变过程"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    logger.info("正在生成 64x64 去噪过程可视化关键帧...")
    ddpm = load_model()
    num_samples = 5  # 展示 5 组样本
    
    # 初始纯噪声
    img = torch.randn((num_samples, 1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
    
    # 定义展示的关键时间节点 (从 T 到 0)
    stages = [1000, 800, 600, 400, 200, 100, 50, 0]
    stages_imgs = {s: [] for s in stages}

    # 逐步去噪
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((num_samples,), i, device=DEVICE, dtype=torch.long)
        img = ddpm.p_sample(img, t, i)
        
        # 记录关键帧
        curr_t = i  # 实际步数
        if curr_t in [s-1 for s in stages] or curr_t == 0:
            # 查找对应的阶段键名
            display_t = 0 if curr_t == 0 else (curr_t + 1)
            if display_t in stages:
                # 转换到 [0, 1] 供绘图
                norm_img = torch.clamp((img + 1) / 2, 0.0, 1.0)
                stages_imgs[display_t] = norm_img.cpu().numpy()

    # 绘图：行代表样本，列代表去噪阶段
    fig, axes = plt.subplots(num_samples, len(stages), figsize=(len(stages)*2, num_samples*2))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for row in range(num_samples):
        for col, s in enumerate(stages):
            ax = axes[row, col]
            # 提取对应阶段的图片
            img_data = stages_imgs[s][row].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            ax.imshow(img_data, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"T={s}", fontsize=10)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_fn = IMAGE_DIR / f"geo_denoise_process_{timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info(f"去噪过程展示图已保存为 {grid_fn.name}")

@torch.no_grad()
def evaluate_model():
    if not LATEST_MODEL_PATH.exists(): return
    logger.info("开始地质图生成质量评估 (FID + P&R)...")
    ddpm = load_model()
    
    total_eval_samples = 200 
    gen_imgs = ddpm.sample(total_eval_samples)
    
    # 获取真实图对比
    real_list = []
    for x, _ in test_loader:
        x_norm = (x + 1) / 2
        real_list.append(x_norm)
        if len(torch.cat(real_list)) >= total_eval_samples: break
    all_real_imgs = torch.cat(real_list)[:total_eval_samples].to(DEVICE)

    fid, prec, recall = compute_fid_and_pr(gen_imgs, all_real_imgs, device=DEVICE)
    
    logger.info(f"评估结果: FID={fid:.4f}, Precision={prec:.4f}, Recall={recall:.4f}")

# Main 菜单逻辑保持不变
def main():
    while True:
        print("\n" + "="*40)
        print("    DDPM 2D 地质图 (64x64) 管理系统")
        print(" [1] 训练模型")
        print(" [2] 生成测试 (9张宫格)")
        print(" [3] 评估模型 (FID/PR)")
        print(" [0] 退出")
        choice = input("请选择: ").strip()
        if choice == '1': train_model()
        elif choice == '2':
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (生成 9 张地质图)")
                logger.info("  [2] Denoising Process (去噪渐变展示)")
                logger.info("  [0/exit] 返回主菜单")

                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == '1':
                    generate_mode()
                elif sub_choice == '2':
                    denoise_process_mode()
                elif sub_choice in ['0', 'exit']:
                    break
        elif choice == '3': evaluate_model()
        elif choice == '0': break

if __name__ == '__main__':
    main()