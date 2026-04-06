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

# --- 路径与环境设置 (保留你的原逻辑) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 为了避免和 VAE 的模型冲突，重命名了保存文件
LATEST_MODEL_PATH = MODEL_DIR / "latest_ddpm_model.pth" 

COMBINE_IMAGES_ROW: int = 3
COMBINE_IMAGES_COL: int = 3
IMAGE_WIDTH: int = 28
IMAGE_HEIGHT: int = 28

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time
from utils.image_evaluator import evaluate_images
from utils.fid_pr_evaluator import compute_fid_and_pr 

# --- 1. 参数设置 ---
DEVICE = init_gpu_environment()
BATCH_SIZE = 128
NUM_EPOCHS = 50  # 扩散模型收敛比VAE慢，建议稍微增加一点 epoch

LR = 2e-4
WEIGHT_DECAY = 1e-4

# 扩散模型超参数
TIMESTEPS = 1000       # 加噪/去噪的总步数
BETA_START = 1e-4      # beta 调度起点
BETA_END = 0.02        # beta 调度终点

# --- 2. 加载 MNIST 数据集 ---
# 注意：对于扩散模型，将图像映射到 [-1, 1] 会让正态分布加噪效果更好
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# --- 3. DDPM 网络核心模块: 带有 Time Embedding 的 U-Net ---

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步 t 的正弦位置编码"""
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
    """U-Net 的基础残差块，融合了时间步信息"""
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
        # 融入时间信息
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  # [B, C, 1, 1]
        h = h + time_emb
        h = self.norm2(self.conv2(h))
        return self.relu(h)

class UNet(nn.Module):
    """预测噪声的 U-Net"""
    def __init__(self):
        super().__init__()
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # 编码器 (Down) 28x28 -> 14x14 -> 7x7
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.down1 = Block(32, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层 (Bottleneck) 7x7
        self.bot1 = Block(128, 128, time_emb_dim)

        # 解码器 (Up) 7x7 -> 14x14 -> 28x28
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 7 -> 14
        self.up_block1 = Block(64 + 64, 64, time_emb_dim) # Concat skip connection
        
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 14 -> 28
        self.up_block2 = Block(32 + 32, 32, time_emb_dim)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        # Down
        x0 = self.conv0(x)       # [B, 32, 28, 28]
        x1 = self.down1(x0, t)   # [B, 64, 28, 28] -> pool -> [B, 64, 14, 14]
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)   # [B, 128, 14, 14] -> pool -> [B, 128, 7, 7]
        p2 = self.pool2(x2)

        # Bottleneck
        bot = self.bot1(p2, t)   # [B, 128, 7, 7]

        # Up (带着 skip connections)
        u1 = self.up1(bot)       # [B, 64, 14, 14]
        u1 = torch.cat([u1, x2], dim=1)  # concat 64 + 128 = 192? No, x2 is 128, p1 is 64. Wait.
        # Let's fix skip connections matching:
        # x0: 32 (28x28), x1: 64 (28x28) -> p1: 64(14x14)
        # x2: 128 (14x14) -> p2: 128(7x7)
        # u1: 64 (14x14), skip is x2 (128)? No, skip should match spatial size. 
        # Actually u1 matches p1 (14x14). So skip is p1 (64).
        pass # Fixed below:

        # Let's redefine forwarding shapes explicitly:
        x_init = self.conv0(x)                   # 28x28, 32
        d1 = self.down1(x_init, t)               # 28x28, 64
        p1 = self.pool1(d1)                      # 14x14, 64
        d2 = self.down2(p1, t)                   # 14x14, 128
        p2 = self.pool2(d2)                      # 7x7, 128

        bot = self.bot1(p2, t)                   # 7x7, 128

        u1 = self.up1(bot)                       # 14x14, 64
        u1 = torch.cat([u1, d2], dim=1)          # 14x14, 64+128=192
        u1 = self.up_block1(u1, t)               # 14x14, 64 (Reinit up_block1 in init)

        u2 = self.up2(u1)                        # 28x28, 32
        u2 = torch.cat([u2, d1], dim=1)          # 28x28, 32+64=96
        u2 = self.up_block2(u2, t)               # 28x28, 32
        
        return self.out(u2)


# 重写 UNet 的 __init__ 让通道对齐
UNet.__init__ = lambda self: UNet_init(self)
def UNet_init(self):
    nn.Module.__init__(self)
    time_emb_dim = 128
    self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.SiLU()
    )
    self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
    self.down1 = Block(32, 64, time_emb_dim)
    self.pool1 = nn.MaxPool2d(2)
    self.down2 = Block(64, 128, time_emb_dim)
    self.pool2 = nn.MaxPool2d(2)

    self.bot1 = Block(128, 128, time_emb_dim)

    self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  
    self.up_block1 = Block(64 + 128, 64, time_emb_dim)  # u1(64) + d2(128)
    
    self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    
    self.up_block2 = Block(32 + 64, 32, time_emb_dim)   # u2(32) + d1(64)

    self.out = nn.Conv2d(32, 1, 1)


# --- 4. 扩散模型框架调度器 DDPM ---
class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # 定义线性调度的 beta 参数
        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向加噪需要用到的常数
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # 逆向去噪需要用到的常数
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def extract(self, a, t, x_shape):
        """辅助函数：从预计算列表中抽取当前 batch 中每个 t 对应的系数"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x_0):
        """训练前向过程：随机挑选 t 加噪，并计算与模型预测噪声的 MSE Loss"""
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        
        # 按照公式 x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise 得到加噪图片
        x_t = (
            self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        
        # U-Net 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 扩散模型核心就是对齐真实噪声和预测噪声的分布 (MSE Loss)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, t_index):
        """逆向的一步去噪"""
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # Equation 11 in the DDPM paper
        # 用模型预测的噪声来修正图像
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * self.model(x_t, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean # 最后一步不加额外的噪声了
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            # 加上郎之万动力学噪声 (方差)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample(self, num_images):
        """完整的生成过程：从纯噪声逐渐降噪 T 步"""
        self.model.eval()
        img = torch.randn((num_images, 1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
        
        # 从 T 倒数到 0
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        # 将 [-1, 1] 映射回 [0, 1] 方便显示
        img = (img + 1) / 2
        img = torch.clamp(img, 0.0, 1.0)
        return img


# --- 5. 训练模型 ---
def train_model():
    unet = UNet().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    optimizer = optim.AdamW(ddpm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    loss_list = []
    start_epoch = 0
    end_epoch = start_epoch + NUM_EPOCHS
    train_times = 0
    total_training_time = 0.0

    start_time = time.time()

    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        start_epoch = checkpoint["epoch"]
        end_epoch = start_epoch + NUM_EPOCHS
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)
        logger.info(f"已加载 DDPM 模型, 起始 epoch={start_epoch}， 终止 epoch={end_epoch}")
    else:
        logger.info("第一次训练 DDPM 模型")

    for epoch in range(start_epoch, end_epoch):
        ddpm.train()
        avg_loss = 0
        total_n = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            bs = x.size(0)

            optimizer.zero_grad(set_to_none=True)
            # 前向计算 Loss
            loss = ddpm(x)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            total_n += bs
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)

        loss_list.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] 扩散模型 MSE Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    total_training_time += elapsed

    torch.save({
        'epoch': end_epoch,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1,
        'total_training_time': total_training_time,
    }, f=LATEST_MODEL_PATH)

    logger.info(f"DDPM 模型已保存，本次耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s")

    # 保存 Loss 图表
    plt.figure(figsize=(6, 4))
    epochs_range = np.arange(1, len(loss_list) + 1)
    plt.plot(epochs_range, loss_list, color='blue', marker='o', alpha=0.7)
    plt.title("DDPM Noise Prediction MSE Loss")
    plt.xlabel("epochs")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"ddpm_loss_{timestamp}_epoch_{end_epoch}.png")
    plt.close()


def show_model_status():
    if not LATEST_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现模型文件。")
        return
    try:
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
        epoch = checkpoint.get('epoch', 0)
        train_times = checkpoint.get('train_times', 0)
        loss_list = checkpoint.get('train_losses', [])
        latest_loss = loss_list[-1] if loss_list else "N/A"
        total_training_time = checkpoint.get('total_training_time', 0)

        logger.info(f"{'DDPM 扩散模型状态报告':^36}")
        logger.info(f" 已训练总轮数:    {epoch}")
        logger.info(f" 累计训练次数:    {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f" 最近 MSE Loss:   {latest_loss:.4f} (越小说明去噪越准确)")
        
        duration = total_training_time / 60
        logger.info(f" 训练总时长：     {duration:.1f} 分钟")
        logger.info("=" * 40 + "\n")
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")


def load_model() -> DDPM:
    unet = UNet().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()
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
    """模式 2-1: 从纯噪声生成 9 张图片"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    logger.info(f"开始 DDPM 采样，共需要降噪 {TIMESTEPS} 步，请稍等...")
    ddpm = load_model()
    current_timestamp = get_current_time()
    
    # 执行缓慢的逆向去噪采样
    num_samples = COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL
    gen_imgs = ddpm.sample(num_samples).cpu().numpy()

    # 1. 存独立图
    for i in range(num_samples):
        filename = IMAGE_DIR / f"ddpm_generate_{current_timestamp}_{i + 1:02d}.png"
        save_large_image(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), filename)

    # 2. 存宫格图
    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
        ax.axis('off')
    grid_fn = IMAGE_DIR / f"ddpm_generate_{current_timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("DDPM Generate 模式运行完毕，图片已保存。")


@torch.no_grad()
def denoise_process_mode():
    """模式 2-2: DDPM 特色功能 —— 展示从噪声到图像的去噪渐变过程"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    logger.info("正在生成去噪过程的可视化...")
    ddpm = load_model()
    num_samples = 5  # 我们展示 5 张图的渐变过程
    
    img = torch.randn((num_samples, 1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
    
    # 我们抽取 8 个关键帧来展示渐变
    stages = [1000, 800, 600, 400, 200, 100, 50, 0]
    stages_imgs = {s: [] for s in stages}

    for i in reversed(range(0, ddpm.timesteps)):
        t = torch.full((num_samples,), i, device=DEVICE, dtype=torch.long)
        img = ddpm.p_sample(img, t, i)
        
        # 记录关键帧
        if (i + 1) in stages or i == 0:
            save_t = i + 1 if i != 0 else 0
            # [-1, 1] -> [0, 1] 截断
            norm_img = torch.clamp((img + 1) / 2, 0.0, 1.0)
            stages_imgs[save_t] = norm_img.cpu().numpy()

    # 画一幅宽图：行是样本，列是去噪步数
    fig, axes = plt.subplots(num_samples, len(stages), figsize=(len(stages)*1.5, num_samples*1.5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for row in range(num_samples):
        for col, s in enumerate(stages):
            ax = axes[row, col]
            ax.imshow(stages_imgs[s][row].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"t={s}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_fn = IMAGE_DIR / f"ddpm_denoise_process_{timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("去噪过程展示图已保存。")


# ==========================================
# [模式 4] 评估 DDPM 生成质量的入口函数 (完整版: IS + FID + P&R)
# ==========================================
@torch.no_grad()
def evaluate_model():
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到 DDPM 模型，请先训练。")
        return

    logger.info("=" * 40)
    logger.info("开始终极评估模式 (Evaluate Mode)...")
    logger.info("将计算 置信度、熵、覆盖率、FID、Precision、Recall 6大指标")
    
    ddpm = load_model()
    
    # 样本总数 (为了 P&R 的流形估计准确，建议样本不低于 250，有条件可以调到 1000)
    total_eval_samples = 250  
    batch_size = 50  
    gen_list = []
    
    start_time = time.time()
    
    # 1. 生成假图片
    for i in range(0, total_eval_samples, batch_size):
        curr_bs = min(batch_size, total_eval_samples - i)
        logger.info(f" -> 正在生成批次 [{i + curr_bs}/{total_eval_samples}]...")
        imgs = ddpm.sample(curr_bs)  
        gen_list.append(imgs)
    all_gen_imgs = torch.cat(gen_list, dim=0)

    # 2. 抽取真图片
    logger.info(" -> 正在从测试集抽取真实图片...")
    real_list = []
    collected = 0
    for x, _ in test_loader:
        x_norm = (x + 1) / 2
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        take = min(x_norm.size(0), total_eval_samples - collected)
        real_list.append(x_norm[:take])
        collected += take
        if collected >= total_eval_samples: break
    all_real_imgs = torch.cat(real_list, dim=0)

    try:
        # 3. 计算基于分类器的指标 (IS家族)
        logger.info(" -> 计算分类器指标 (置信度/熵/覆盖率)...")
        max_conf, entropy, coverage = evaluate_images(all_gen_imgs, device=DEVICE)
        
        # 4. 计算基于特征空间的指标 (FID, P&R)
        resnet_path = MODEL_DIR / 'resnet18-f37072fd.pth'
        r_path_str = str(resnet_path) if resnet_path.exists() else None

        logger.info(" -> 计算特征流形指标 (FID / Precision / Recall)...")
        fid_score, precision, recall = compute_fid_and_pr(
            gen_images=all_gen_imgs, 
            real_images=all_real_imgs, 
            device=DEVICE, 
            batch_size=128
        )

        elapsed = time.time() - start_time
        
        # 5. 打印六边形战士评测战报
        logger.info("\n" + "=" * 55)
        logger.info(f"{'DDPM 终极生成质量评估报告':^50}")
        logger.info("-" * 55)
        logger.info(f" 评估规模     : {total_eval_samples} 生成 vs {total_eval_samples} 真实")
        logger.info(f" 评测耗时     : {elapsed:.2f} 秒")
        logger.info("-" * 55)
        logger.info(f" [分类微观指标 - 基于 MNIST 分类器]")
        logger.info(f" 平均置信度   : {max_conf:.4f}  (越高越好，数字是否逼真可认)")
        logger.info(f" 类别覆盖率   : {coverage:.4f}  (越高越好，0-9类别是否均匀)")
        logger.info(f" 平均熵值     : {entropy:.4f}  (越低越好，分类器判断是否坚决)")
        logger.info("-" * 55)
        logger.info(f" [流形宏观指标 - 基于 ResNet18 特征空间]")
        logger.info(f" FID 分数     : {fid_score:.4f}  (越低越好，整体风格距离)")
        logger.info(f" Precision    : {precision:.4f}  (越高越好，保真度/画工质量)")
        logger.info(f" Recall       : {recall:.4f}  (越高越好，多样性/是否漏图)")
        logger.info("=" * 55 + "\n")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")

def main():
    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      DDPM 扩散模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Denoising)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 评估模型指标(Evaluate)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == '1':
            train_model()
        elif choice == '2':
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (从纯噪声逆向采样生成 9 图)")
                logger.info("  [2] Denoising Process (展示 5 张图逐渐去噪的帧序列)")
                logger.info("  [0/exit] 返回主菜单")

                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == '1':
                    generate_mode()
                elif sub_choice == '2':
                    denoise_process_mode()
                elif sub_choice in ['0', 'exit']:
                    break
                else:
                    logger.info("  无效输入。")
        elif choice == '3':
            show_model_status()
        elif choice == '4':
            evaluate_model()
        elif choice in ['0', 'exit']:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3 或 0")


if __name__ == '__main__':
    main()