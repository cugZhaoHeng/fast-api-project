import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import copy
from scipy import linalg
from mnist_classifier import load_classifier, evaluate_generated_images

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
MNIST_DIR = PROJECT_ROOT_DIR / 'data' / 'MNIST'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
from utils.fid_evaluator import compute_fid

# --- 1. 参数设置 ---
device = init_gpu_environment()
batch_size = 128
NUM_EPOCHS = 500

LR = 1e-3
WEIGHT_DECAY = 1e-5
LATENT_DIM = 24

BETA_MAX = 0.15              # 降低KL压力，避免过分“平均化”
WARMUP_EPOCHS = 40           # 更长warmup
FREE_BITS_PER_DIM = 0.02     # KL free bits
L1_WEIGHT = 0.15             # 辅助边缘清晰（不要太大，防止失真）
EMA_DECAY = 0.999

NUM_WORKERS = 0              # Windows稳定优先
PIN_MEMORY = torch.cuda.is_available()

N_GEN_EVAL = 10000
GEN_BATCH = 256

# --- 2. 加载 MNIST 数据集 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# =========================
# Model
# =========================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ConvVAE(nn.Module):
    def __init__(self, z_dim=24):
        super().__init__()
        self.z_dim = z_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28 -> 14
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),

            nn.Conv2d(32, 64, 4, 2, 1),  # 14 -> 7
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResBlock(64),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)

        # Decoder
        self.fc_dec = nn.Linear(z_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)    # 14 -> 28 (logits)
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -8.0, 6.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


# =========================
# Loss & Schedules
# =========================
def beta_schedule(epoch):
    if epoch >= WARMUP_EPOCHS:
        return BETA_MAX
    return BETA_MAX * (epoch / max(1, WARMUP_EPOCHS))


def loss_fn(logits, x, mu, logvar, beta=1.0, free_bits_per_dim=0.0, l1_weight=0.0):
    # BCE recon per sample
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none').flatten(1).sum(1)
    # L1 recon per sample
    x_hat = torch.sigmoid(logits)
    l1 = F.l1_loss(x_hat, x, reduction='none').flatten(1).sum(1)

    # KL per dim -> free bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    if free_bits_per_dim > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits_per_dim)
    kl = kl_per_dim.sum(1)  # [B]

    recon = bce + l1_weight * l1
    elbo = recon + beta * kl
    loss = elbo.mean()
    # logger.info(f"recon shape:{recon.shape}, kl shape:{kl.shape}")
    return loss, recon.mean(), kl.mean()


# --- 5. 训练模型 ---
model = ConvVAE(z_dim=LATENT_DIM).to(device)
ema = EMA(model, decay=EMA_DECAY)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

model.train()


def train_model():
    loss_list, bce_list, kld_list = [], [], []
    start_epoch, train_times = 0, 0
    total_training_time = 0.0  # 累计训练总时长（秒）
    
    # 记录本次运行开始时间
    start_time = time.time()
    
    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        bce_list = checkpoint.get("bce_losses", [])
        kld_list = checkpoint.get("kld_losses", [])
        start_epoch = checkpoint["epoch"]
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)  # 恢复总时长
        logger.info(f"已加载模型, 起始 epoch={start_epoch}, 累计训练时长={total_training_time:.2f}s")
    else:
        logger.info("第一次训练模型")

    model.train()
    end_epoch = start_epoch + NUM_EPOCHS
    
    for epoch in range(start_epoch, end_epoch):
        avg_loss, avg_bce, avg_kld = 0, 0, 0
        temp_mu, temp_std, temp_logvar = [], [], []
        
        beta = beta_schedule(epoch)
        total_loss, total_n = 0.0, 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            bs = x.size(0)

            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x)
            loss, bce, kld = loss_fn(
                logits, x, mu, logvar,
                beta=beta,
                free_bits_per_dim=FREE_BITS_PER_DIM,
                l1_weight=L1_WEIGHT
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item() * bs
            total_n += bs
            
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            avg_bce = (avg_bce * batch_idx + bce.item()) / (batch_idx + 1)
            avg_kld = (avg_kld * batch_idx + kld.item()) / (batch_idx + 1)
            
            temp_mu.append(mu.mean().item())
            temp_std.append(mu.std().item())
            temp_logvar.append(logvar.mean().item())

        loss_list.append(avg_loss)
        bce_list.append(avg_bce)
        kld_list.append(avg_kld)
        
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f}), beta: {beta}")

    # 计算本次运行耗时
    elapsed = time.time() - start_time
    total_training_time += elapsed
    
    # 保存模型
    torch.save({
        'epoch': end_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'bce_losses': bce_list,
        'kld_losses': kld_list,
        'train_times': train_times + 1,
        'total_training_time': total_training_time,  # 添加累计训练总时长
        'mu_mean': float(np.mean(temp_mu)),
        'mu_std': float(np.mean(temp_std)),
        'logvar_mean': float(np.mean(temp_logvar))
    }, f=LATEST_MODEL_PATH)
    
    logger.info(f"模型已保存，本次训练耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s")


    plt.figure(figsize=(15, 5))
    titles = ['Total Loss', 'BCE (Recon Loss)', 'KLD (KL Loss)']
    data_to_plot = [loss_list, bce_list, kld_list]
    colors = ['b', 'g', 'r']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        curr_data = data_to_plot[i]
        epochs_range = np.arange(1, len(curr_data) + 1)
        plt.plot(epochs_range, curr_data, color=colors[i], alpha=0.3)
        
        # 20点采样逻辑
        num_pts = min(len(curr_data), 20)
        indices = np.linspace(0, len(curr_data)-1, num_pts, dtype=int)
        plt.scatter(epochs_range[indices], np.array(curr_data)[indices], color=colors[i], s=30)
        
        plt.title(titles[i])
        plt.xlabel("epochs")
        plt.grid(True, linestyle='--', alpha=0.5)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"loss_metrics_{timestamp}_epoch_{end_epoch}.png")
    plt.close()
    logger.info("指标图表已保存。")

def show_model_status():
    """查看当前模型参数及隐空间分布状态"""
    if not LATEST_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现模型文件。")
        return

    try:
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
        
        epoch = checkpoint.get('epoch', 0)
        train_times = checkpoint.get('train_times', 0)
        loss_list = checkpoint.get('train_losses', [])
        latest_loss = loss_list[-1] if loss_list else "N/A"
        total_training_time = checkpoint.get('total_training_time', 0)
        
        # 提取隐空间统计量
        mu_mean = checkpoint.get('mu_mean', "N/A")
        mu_std = checkpoint.get('mu_std', "N/A")
        logvar_mean = checkpoint.get('logvar_mean', "N/A")

        logger.info(f"{'模型状态报告':^36}")
        logger.info(f" 已训练总轮数:    {epoch}")
        logger.info(f" 累计训练次数:    {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f" 最近平均 Loss:   {latest_loss:.4f}")
        # 格式化训练时长
        if total_training_time < 3600:  # 小于1小时
            duration = total_training_time / 60
            unit = "分钟"
        else:
            duration = total_training_time / 3600
            unit = "小时"

        logger.info(f" 训练时长：{duration:.1f}{unit}")
        
        logger.info(f"{'隐空间分布 (Latent Space Check)':^36}")
        # 核心调试指标
        if isinstance(mu_mean, (float, int)):
            logger.info(f" Mu 均值 (应接近 0):   {mu_mean:+.4f}")
            logger.info(f" Mu 标准差 (应接近 1): {mu_std:.4f}")
            logger.info(f" LogVar 均值 (应负数): {logvar_mean:.4f}")
            
            # 简单的自动诊断
            if abs(mu_mean) > 0.5 or abs(mu_std - 1.0) > 0.5:
                logger.info("\n[诊断结论]: 隐空间未对齐标准正态分布。")
                logger.info(" -> 原因: KLD 权重可能太低 (当前 0.1)。")
                logger.info(" -> 结果: Generate 模式无法生成有效数字。")
            else:
                logger.info("\n[诊断结论]: 隐空间分布良好。")
        else:
            logger.info(" 暂无隐空间统计数据，请先执行一次训练。")
        logger.info("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")

def save_large_image(img_data, path, title=None, is_compare=False):
    """保存大尺寸图片的辅助函数"""
    # 如果是对比图，宽度加倍
    figsize = (8, 4) if is_compare else (4, 4)
    plt.figure(figsize=figsize, dpi=100) # 100 DPI 下 4英寸=400像素
    
    if is_compare:
        # img_data 预期为 (28, 56) 的拼接图
        plt.imshow(img_data, cmap='gray')
    else:
        plt.imshow(img_data, cmap='gray')
        
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def generate_mode():
    """模式 2-1: 生成 9 张大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ema = EMA(model, decay=EMA_DECAY)
    eval_model = ema.shadow
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with torch.no_grad():
        z = torch.randn(9, LATENT_DIM, device=device)
        gen_imgs = torch.sigmoid(eval_model.decode_logits(z)).cpu()
        

        # 1. 保存 9 张独立大图
        for i in range(9):
            # 命名: vae_image_generate_时间戳_编号.png
            filename = IMAGE_DIR / f"vae_image_generate_{timestamp}_{i+1:02d}.png"
            save_large_image(gen_imgs[i].reshape(28, 28), filename)
        
        # 2. 保存 3*3 宫格图
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(gen_imgs[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
        grid_fn = IMAGE_DIR / f"vae_image_generate_{timestamp}.png"
        plt.savefig(grid_fn, bbox_inches='tight')
        plt.close()
        logger.info("Generate 模式运行完毕，图片已保存。")

def compare_mode():
    """模式 2-2: 重构 9 张对比大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ema = EMA(model, decay=EMA_DECAY)
    eval_model = ema.shadow

    loader = DataLoader(datasets.MNIST(root=DATA_DIR, train=False, transform=transform), batch_size=9, shuffle=True)
    
    # 收集原始图像和对应的生成图像
    orig_list = []   # 存放原始图像张量
    gen_list = []    # 存放生成图像张量
    need = 9
    it = iter(loader)
    
    while need > 0:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x, _ = next(it)
        x = x.to(device, non_blocking=True)
        take = min(need, x.size(0))
        orig_list.append(x[:take])
        
        # 对这批图像编码后解码（使用聚合后验采样）
        mu, logvar = eval_model.encode(x[:take])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        gen = torch.sigmoid(eval_model.decode_logits(z)).cpu()
        gen_list.append(gen)
        
        need -= take

    # 合并所有批次
    orig = torch.cat(orig_list, dim=0).cpu()   # [9,1,28,28]
    gen = torch.cat(gen_list, dim=0)           # [9,1,28,28]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with torch.no_grad():
        # 转为 numpy 并去除通道维度
        orig_np = orig.squeeze().numpy()        # [9,28,28]
        recon_np = gen.squeeze().numpy()        # [9,28,28]

        # 1. 保存 9 张独立对比大图
        for i in range(9):
            # 左右拼接
            combined = np.hstack((orig_np[i], recon_np[i]))
            filename = IMAGE_DIR / f"vae_image_compare_{timestamp}_{i+1:02d}.png"
            save_large_image(combined, filename, title="Original | Reconstructed", is_compare=True)

        # 2. 保存 3*3 宫格对比总图
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        for i, ax in enumerate(axes.flat):
            combined = np.hstack((orig_np[i], recon_np[i]))
            ax.imshow(combined, cmap='gray')
            ax.set_title(f"Pair {i+1:02d}")
            ax.axis('off')
        grid_fn = IMAGE_DIR / f"vae_image_compare_{timestamp}.png"
        plt.savefig(grid_fn, bbox_inches='tight')
        plt.close()
        logger.info("Compare 模式运行完毕，图片已保存。")

def generate_from_posterior_mode():
    """从后验聚合分布采样生成图像"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ema = EMA(model, decay=EMA_DECAY)
    eval_model = ema.shadow

    # 收集所有训练样本的 mu（或 z）
    all_mu = []
    with torch.no_grad():
        for x, _ in train_loader:   # 使用训练集 DataLoader
            x = x.to(device, non_blocking=True)
            mu, _ = eval_model.encode(x)   # 只取 mu
            all_mu.append(mu.cpu())
    
    all_mu = torch.cat(all_mu, dim=0)      # [N, LATENT_DIM]
    logger.info(f"已收集 {all_mu.shape[0]} 个样本的 mu")

    # 方法1：直接使用经验分布（从这些 mu 中随机选择一个）
    # 方法2：拟合高斯分布，计算均值向量和协方差矩阵（这里使用对角协方差简化）
    mu_mean = all_mu.mean(dim=0)          # 均值
    mu_cov = all_mu.var(dim=0)            # 对角方差
    # 注意：若需要全协方差，可用 np.cov(all_mu.T)，但计算量大

    # 从拟合的高斯分布中采样
    with torch.no_grad():
        # 生成9个样本
        z = torch.randn(9, LATENT_DIM, device=device) * torch.sqrt(mu_cov).to(device) + mu_mean.to(device)
        gen_imgs = torch.sigmoid(eval_model.decode_logits(z)).cpu()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 9 张大图和宫格图（沿用原有保存逻辑）
    for i in range(9):
        filename = IMAGE_DIR / f"vae_image_posterior_{timestamp}_{i+1:02d}.png"
        save_large_image(gen_imgs[i].reshape(28, 28), filename)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    grid_fn = IMAGE_DIR / f"vae_image_posterior_{timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("从后验聚合分布生成完成。")

def evaluate_model():
    """评估当前 VAE 模型的好坏（基于分类器和 FID）"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型，请先训练。")
        return

    # 加载 VAE 模型和 EMA 模型
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ema = EMA(model, decay=EMA_DECAY)
    eval_model = ema.shadow

    # 确保分类器存在（训练或加载）
    classifier_path = MODEL_DIR / "mnist_classifier.pth"
    if not classifier_path.exists():
        logger.info("分类器不存在，开始训练...")
        from mnist_classifier import train_mnist_classifier  # 导入训练函数
        train_mnist_classifier(train_loader, test_loader, epochs=10, lr=1e-3, save_path=classifier_path)

    # --- 准备真实图像（测试集全部图像，共10000张）---
    real_imgs_list = []
    for x, _ in test_loader:
        real_imgs_list.append(x)
    real_imgs = torch.cat(real_imgs_list, dim=0)[:N_GEN_EVAL]   # 取 N_GEN_EVAL 张（与生成数量一致）
    logger.info(f"已收集 {len(real_imgs)} 张真实图像")

    # ========== 评估标准正态采样 ==========
    logger.info("开始评估标准正态采样...")
    n_gen = N_GEN_EVAL
    batch_size = GEN_BATCH
    all_imgs_norm = []
    with torch.no_grad():
        remain = n_gen
        while remain > 0:
            bs = min(batch_size, remain)
            z = torch.randn(bs, LATENT_DIM, device=device)
            logits = eval_model.decode_logits(z)
            imgs = torch.sigmoid(logits)  # [bs,1,28,28]
            all_imgs_norm.append(imgs.cpu())
            remain -= bs
    gen_imgs_norm = torch.cat(all_imgs_norm, dim=0)  # [n_gen,1,28,28]

    # 分类器评估
    from utils.image_evaluator import evaluate_images   # 之前解耦的评估函数
    conf_n, entropy_n, coverage_n = evaluate_images(gen_imgs_norm, device=device)
    entropy_norm_n = min(entropy_n / 2.3026, 1.0)
    score_n = 100.0 * (0.5 * conf_n + 0.3 * coverage_n + 0.2 * (1.0 - entropy_norm_n))
    logger.info(f"标准正态采样 | 置信度: {conf_n:.4f} | 熵: {entropy_n:.4f} | 覆盖率: {coverage_n:.4f} | 综合得分: {score_n:.2f}")

    # FID 评估
    fid_n = compute_fid(gen_imgs_norm, real_imgs, device=device)
    logger.info(f"标准正态采样 FID: {fid_n:.2f}")

    # ========== 评估聚合后验采样 ==========
    logger.info("开始评估聚合后验采样...")
    all_imgs_agg = []
    with torch.no_grad():
        remain = n_gen
        # 从训练集中采样 mu 和 logvar
        ref_iter = iter(train_loader)
        while remain > 0:
            bs = min(batch_size, remain)
            try:
                x_ref, _ = next(ref_iter)
            except StopIteration:
                ref_iter = iter(train_loader)
                x_ref, _ = next(ref_iter)
            x_ref = x_ref.to(device)
            mu, logvar = eval_model.encode(x_ref)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            logits = eval_model.decode_logits(z)
            imgs = torch.sigmoid(logits)
            all_imgs_agg.append(imgs.cpu())
            remain -= bs
    gen_imgs_agg = torch.cat(all_imgs_agg, dim=0)[:n_gen]  # 确保数量准确

    # 分类器评估
    conf_a, entropy_a, coverage_a = evaluate_images(gen_imgs_agg, device=device)
    entropy_norm_a = min(entropy_a / 2.3026, 1.0)
    score_a = 100.0 * (0.5 * conf_a + 0.3 * coverage_a + 0.2 * (1.0 - entropy_norm_a))
    logger.info(f"聚合后验采样 | 置信度: {conf_a:.4f} | 熵: {entropy_a:.4f} | 覆盖率: {coverage_a:.4f} | 综合得分: {score_a:.2f}")

    # FID 评估
    fid_a = compute_fid(gen_imgs_agg, real_imgs, device=device)
    logger.info(f"聚合后验采样 FID: {fid_a:.2f}")

    # ========== 保存评估结果到文件 ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = MODEL_DIR / f"eval_{timestamp}.txt"
    with open(result_file, 'w') as f:
        f.write("Standard Normal:\n")
        f.write(f"  Confidence: {conf_n:.4f}\n")
        f.write(f"  Entropy: {entropy_n:.4f}\n")
        f.write(f"  Coverage: {coverage_n:.4f}\n")
        f.write(f"  Final Score: {score_n:.2f}\n")
        f.write(f"  FID: {fid_n:.2f}\n\n")
        f.write("Aggregated Posterior:\n")
        f.write(f"  Confidence: {conf_a:.4f}\n")
        f.write(f"  Entropy: {entropy_a:.4f}\n")
        f.write(f"  Coverage: {coverage_a:.4f}\n")
        f.write(f"  Final Score: {score_a:.2f}\n")
        f.write(f"  FID: {fid_a:.2f}\n")
    logger.info(f"评估结果已保存至 {result_file}")

def main():
    while True:
        logger.info("\n" + "="*40)
        logger.info("      VAE 模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Compare)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 评估模型 (Evaluate)")
        logger.info(" [0/exit] 退出程序")
        logger.info("="*40)
        
        choice = input("请选择主菜单功能: ").strip().lower()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (从先验采样)")
                logger.info("  [2] Compare  (重构 9 张测试图片对比)")
                logger.info("  [3] Generate from posterior (从后验聚合分布采样)")
                logger.info("  [0/exit] 返回主菜单并完全退出")
                
                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == '1':
                    generate_mode()
                elif sub_choice == '2':
                    compare_mode()
                elif sub_choice == '3':
                    generate_from_posterior_mode()
                elif sub_choice in ['0', 'exit']:
                    logger.info("退出程序...")
                    sys.exit(0)
                else:
                    logger.info("  无效输入，请输入 1, 2, 3 或 0")
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
