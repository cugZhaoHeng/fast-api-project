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
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 预训练的 VAE 模型路径 和 新的 LDM 模型路径
VAE_MODEL_PATH = MODEL_DIR / "latest_model.pth"
LDM_MODEL_PATH = MODEL_DIR / "latest_ldm_model.pth"

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
NUM_EPOCHS = 10     # 因为在隐空间(24维)用MLP，训练极快，Epoch可以适当设大
LR = 1e-3           # MLP 学习率可以稍微大一点
WEIGHT_DECAY = 1e-4

LATENT_DIM = 24     # 必须与你的 VAE 隐空间维度一致

# 扩散模型超参数
TIMESTEPS = 1000       
BETA_START = 1e-4      
BETA_END = 0.02        

# --- 2. 加载 MNIST 数据集 (注意：对齐 VAE 的 [0, 1] 输入，不使用 -1 到 1 归一化) ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ==========================================
# 3. 完美复刻原有的 VAE 网络结构 (用于加载权重)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch)
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.net(x))

class ConvVAE(nn.Module):
    def __init__(self, z_dim=24):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(inplace=True), ResBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(inplace=True), ResBlock(64),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_dec = nn.Linear(z_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )
    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -8.0, 6.0)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode_logits(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

def load_pretrained_vae():
    """加载已经训练好的 VAE，并冻结其梯度"""
    if not VAE_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到预训练的 VAE 模型: {VAE_MODEL_PATH}\n请先运行 VAE 训练脚本。")
    vae = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(VAE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


# ==========================================
# 4. LDM 的核心：1D 潜空间 MLP 降噪器
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class TimeConditionedMLPBlock(nn.Module):
    """带时间注入的残差 MLP 块"""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        # 注入时间特征
        h = self.act(self.fc1(h) + self.time_proj(t_emb))
        h = self.fc2(self.norm2(h))
        return x + h  # 残差连接

class LatentDenoiserMLP(nn.Module):
    """专门用来处理 1D 隐变量的扩散网络 (替代 U-Net)"""
    def __init__(self, latent_dim=24, hidden_dim=512, num_layers=6):
        super().__init__()
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TimeConditionedMLPBlock(hidden_dim, time_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, timestep):
        t_emb = self.time_mlp(timestep)
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)


# ==========================================
# 5. 扩散框架调度器 DDPM (适配一维 Latent)
# ==========================================
class LatentDDPM(nn.Module):
    def __init__(self, denoiser_model, timesteps=1000):
        super().__init__()
        self.model = denoiser_model
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

    def forward(self, z_0):
        """前向加噪与预测"""
        t = torch.randint(0, self.timesteps, (z_0.shape[0],), device=z_0.device).long()
        noise = torch.randn_like(z_0)
        
        z_t = (
            self.extract(self.sqrt_alphas_cumprod, t, z_0.shape) * z_0 +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape) * noise
        )
        
        predicted_noise = self.model(z_t, t)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, z_t, t, t_index):
        betas_t = self.extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, z_t.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, z_t.shape)
        
        model_mean = sqrt_recip_alphas_t * (
            z_t - betas_t * self.model(z_t, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, z_t.shape)
            noise = torch.randn_like(z_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample_latent(self, num_samples):
        """生成隐变量 z_0"""
        self.model.eval()
        z = torch.randn((num_samples, LATENT_DIM), device=DEVICE)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=DEVICE, dtype=torch.long)
            z = self.p_sample(z, t, i)
        return z


# ==========================================
# 6. 训练与系统控制
# ==========================================
def train_model():
    try:
        vae = load_pretrained_vae()
        logger.info("成功加载预训练的 VAE 权重作为 LDM 的特征提取器。")
    except Exception as e:
        logger.error(str(e))
        return

    mlp_denoiser = LatentDenoiserMLP(latent_dim=LATENT_DIM).to(DEVICE)
    ldm = LatentDDPM(mlp_denoiser, timesteps=TIMESTEPS).to(DEVICE)
    optimizer = optim.AdamW(ldm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    loss_list = []
    start_epoch = 0
    end_epoch = start_epoch + NUM_EPOCHS
    train_times = 0
    total_training_time = 0.0

    start_time = time.time()

    if LDM_MODEL_PATH.exists():
        checkpoint = torch.load(LDM_MODEL_PATH, map_location=DEVICE, weights_only=False)
        ldm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        start_epoch = checkpoint["epoch"]
        end_epoch = start_epoch + NUM_EPOCHS
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)
        logger.info(f"已加载 LDM 模型, 起始 epoch={start_epoch}， 终止 epoch={end_epoch}")
    else:
        logger.info("第一次训练 LDM (Latent Diffusion) 模型")

    for epoch in range(start_epoch, end_epoch):
        ldm.train()
        avg_loss = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            
            # 第一步：用 VAE 将图像压缩进隐空间 (不计算 VAE 梯度)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                # 使用均值 mu 加上再参数化的噪声 z 来训练，增强泛化
                z_0 = vae.reparam(mu, logvar)

            # 第二步：在隐空间上训练 DDPM
            optimizer.zero_grad(set_to_none=True)
            loss = ldm(z_0)
            loss.backward()
            nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
            optimizer.step()

            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)

        loss_list.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] LDM 隐空间噪声预测 MSE Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    total_training_time += elapsed

    torch.save({
        'epoch': end_epoch,
        'model_state_dict': ldm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1,
        'total_training_time': total_training_time,
    }, f=LDM_MODEL_PATH)

    logger.info(f"LDM 模型已保存，本次耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s")


def show_model_status():
    if not LDM_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现 LDM 模型文件。")
        return
    try:
        checkpoint = torch.load(LDM_MODEL_PATH, map_location=DEVICE)
        epoch = checkpoint.get('epoch', 0)
        train_times = checkpoint.get('train_times', 0)
        loss_list = checkpoint.get('train_losses', [])
        latest_loss = loss_list[-1] if loss_list else "N/A"
        total_training_time = checkpoint.get('total_training_time', 0)

        logger.info(f"{'LDM 潜在扩散模型状态报告':^36}")
        logger.info(f" 挂载 VAE 状态:  {'已连接 (维度 '+str(LATENT_DIM)+')'}")
        logger.info(f" 已训练总轮数:    {epoch}")
        logger.info(f" 累计训练次数:    {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f" 最近 MSE Loss:   {latest_loss:.4f}")
        logger.info(f" 训练总时长：     {total_training_time / 60:.1f} 分钟")
        logger.info("=" * 40 + "\n")
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")


def get_pipeline():
    """获取完整的 VAE + LDM 推理流水线"""
    vae = load_pretrained_vae()
    mlp = LatentDenoiserMLP(latent_dim=LATENT_DIM).to(DEVICE)
    ldm = LatentDDPM(mlp, timesteps=TIMESTEPS).to(DEVICE)
    checkpoint = torch.load(LDM_MODEL_PATH, map_location=DEVICE)
    ldm.load_state_dict(checkpoint['model_state_dict'])
    ldm.eval()
    return vae, ldm


def save_large_image(img_data, path: Path, title: str = None):
    plt.figure(figsize=(4, 4), dpi=100)
    plt.imshow(img_data, cmap='gray')
    if title: plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate_mode():
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型。")
        return

    logger.info("启动 VAE + LDM 联合生成流水线...")
    vae, ldm = get_pipeline()
    current_timestamp = get_current_time()
    
    num_samples = COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL
    # 1. LDM 从纯噪声生成高质量隐变量 z
    z_gen = ldm.sample_latent(num_samples)
    # 2. VAE 解码器将 z 变回像素图像
    gen_imgs = torch.sigmoid(vae.decode_logits(z_gen)).cpu().numpy()

    for i in range(num_samples):
        filename = IMAGE_DIR / f"ldm_generate_{current_timestamp}_{i + 1:02d}.png"
        save_large_image(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), filename)

    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(6, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
        ax.axis('off')
    grid_fn = IMAGE_DIR / f"ldm_generate_{current_timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("LDM Generate 模式运行完毕，图片已保存。")


@torch.no_grad()
def denoise_process_mode():
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型。")
        return

    logger.info("展示潜空间去噪并映射到像素空间的过程...")
    vae, ldm = get_pipeline()
    num_samples = 5  
    
    # 初始化隐变量噪声
    z_img = torch.randn((num_samples, LATENT_DIM), device=DEVICE)
    stages = [1000, 800, 600, 400, 200, 100, 50, 0]
    stages_imgs = {s: [] for s in stages}

    for i in reversed(range(0, ldm.timesteps)):
        t = torch.full((num_samples,), i, device=DEVICE, dtype=torch.long)
        z_img = ldm.p_sample(z_img, t, i)
        
        if (i + 1) in stages or i == 0:
            save_t = i + 1 if i != 0 else 0
            # 实时用 VAE 解码当前的隐变量状态
            decoded_pixel = torch.sigmoid(vae.decode_logits(z_img))
            stages_imgs[save_t] = decoded_pixel.cpu().numpy()

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
    grid_fn = IMAGE_DIR / f"ldm_denoise_process_{timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("LDM 去噪过程展示图已保存。")


@torch.no_grad()
def evaluate_model():
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型。")
        return

    logger.info("=" * 40)
    logger.info("开始终极评估模式 (LDM Evaluate Mode)...")
    
    vae, ldm = get_pipeline()
    total_eval_samples = 2000  # 建议保留 2000 以获得稳定结果
    batch_size = 200           # 因为 LDM(MLP) 推理极快，batch 可以开大点
    gen_list = []
    
    start_time = time.time()
    
    # 1. 快速流水线生成
    for i in range(0, total_eval_samples, batch_size):
        curr_bs = min(batch_size, total_eval_samples - i)
        logger.info(f" -> LDM流水线生成批次 [{i + curr_bs}/{total_eval_samples}]...")
        # LDM 生成潜变量
        z_gen = ldm.sample_latent(curr_bs)
        # VAE 解码为图片 (已在 [0, 1] 范围)
        imgs = torch.sigmoid(vae.decode_logits(z_gen))
        gen_list.append(imgs)
        
    all_gen_imgs = torch.cat(gen_list, dim=0)

    # 2. 抽取真实图片
    logger.info(" -> 正在从测试集抽取真实图片...")
    real_list = []
    collected = 0
    for x, _ in test_loader:
        # VAE 方案中，x 本身就是 [0, 1]，无需做归一化平移
        take = min(x.size(0), total_eval_samples - collected)
        real_list.append(x[:take].to(DEVICE))
        collected += take
        if collected >= total_eval_samples: break
    all_real_imgs = torch.cat(real_list, dim=0)

    try:
        logger.info(" -> 计算分类器指标 (置信度/熵/覆盖率)...")
        max_conf, entropy, coverage = evaluate_images(all_gen_imgs, device=DEVICE)
        
        logger.info(" -> 计算特征流形指标 (FID / Precision / Recall)...")
        fid_score, precision, recall = compute_fid_and_pr(
            gen_images=all_gen_imgs, 
            real_images=all_real_imgs, 
            device=DEVICE, 
            batch_size=128
        )

        elapsed = time.time() - start_time
        
        logger.info("\n" + "=" * 55)
        logger.info(f"{'LDM 联合架构 生成质量评估报告':^50}")
        logger.info("-" * 55)
        logger.info(f" 评估规模     : {total_eval_samples} 生成 vs {total_eval_samples} 真实")
        logger.info(f" 评测耗时     : {elapsed:.2f} 秒")
        logger.info("-" * 55)
        logger.info(f" [分类微观指标 - 基于 MNIST 分类器]")
        logger.info(f" 平均置信度   : {max_conf:.4f}")
        logger.info(f" 类别覆盖率   : {coverage:.4f}")
        logger.info(f" 平均熵值     : {entropy:.4f}")
        logger.info("-" * 55)
        logger.info(f" [流形宏观指标 - 基于 ResNet18 特征空间]")
        logger.info(f" FID 分数     : {fid_score:.4f}  (比单 VAE 有显著提升)")
        logger.info(f" Precision    : {precision:.4f}")
        logger.info(f" Recall       : {recall:.4f}")
        logger.info("=" * 55 + "\n")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")


def main():
    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      LDM 潜在扩散模型管理系统")
        logger.info(" [1] 训练模型 (Train LDM in VAE Latent Space)")
        logger.info(" [2] 测试模式 (Test: Generate/Denoising)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 评估模型生成质量 (Evaluate)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == '1':
            train_model()
        elif choice == '2':
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (用 VAE + LDM 流水线生成图片)")
                logger.info("  [2] Denoising Process (展示潜空间去噪过程)")
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
            logger.info("无效输入，请重新输入 1, 2, 3, 4 或 0")

if __name__ == '__main__':
    main()