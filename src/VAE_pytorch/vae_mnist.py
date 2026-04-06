import os
import sys
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

LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"
COMBINE_IMAGES_ROW: int = 3
COMBINE_IMAGES_COL: int = 3
IMAGE_WIDTH: int = 28
IMAGE_HEIGHT: int = 28

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time
# 引入评估工具 (请确保路径与 DDPM 中一致)
from utils.image_evaluator import evaluate_images
from utils.fid_pr_evaluator import compute_fid_and_pr

logger = create_logger(__name__)

# --- 1. 参数设置 ---
DEVICE = init_gpu_environment()
BATCH_SIZE = 128
NUM_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5
LATENT_DIM = 24

# --- 2. 数据加载 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 模型定义 ---
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
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1),
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
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
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

def beta_schedule(epoch):
    if epoch >= 40: return 0.15
    return 0.15 * (epoch / max(1, 40))

def loss_fn(logits, x, mu, logvar, beta=1.0, free_bits_per_dim=0.0, l1_weight=0.0):
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none').flatten(1).sum(1)
    x_hat = torch.sigmoid(logits)
    l1 = F.l1_loss(x_hat, x, reduction='none').flatten(1).sum(1)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits_per_dim > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits_per_dim)
    kl = kl_per_dim.sum(1)
    recon = bce + l1_weight * l1
    elbo = recon + beta * kl
    return elbo.mean(), recon.mean(), kl.mean()

# --- 5. 核心功能函数 ---

def train_model():
    model = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    loss_list, bce_list, kld_list = [], [], []
    start_epoch = 0
    train_times = 0
    total_training_time = 0.0
    start_time = time.time()

    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        bce_list = checkpoint.get("bce_losses", [])
        kld_list = checkpoint.get("kld_losses", [])
        start_epoch = checkpoint["epoch"]
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)
        logger.info(f"已加载模型, 起始 epoch={start_epoch}, 累计时长={total_training_time:.2f}s")
    
    end_epoch = start_epoch + NUM_EPOCHS
    for epoch in range(start_epoch, end_epoch):
        model.train()
        avg_loss, avg_bce, avg_kld = 0, 0, 0
        temp_mu, temp_std, temp_logvar = [], [], []
        beta = beta_schedule(epoch)

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x)
            loss, bce, kld = loss_fn(logits, x, mu, logvar, beta=beta, free_bits_per_dim=0.02, l1_weight=0.15)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            avg_bce = (avg_bce * batch_idx + bce.item()) / (batch_idx + 1)
            avg_kld = (avg_kld * batch_idx + kld.item()) / (batch_idx + 1)
            temp_mu.append(mu.mean().item())
            temp_std.append(mu.std().item())
            temp_logvar.append(logvar.mean().item())

        loss_list.append(avg_loss)
        bce_list.append(avg_bce)
        kld_list.append(avg_kld)
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] Loss: {avg_loss:.2f} (KLD: {avg_kld:.2f})")

    elapsed = time.time() - start_time
    total_training_time += elapsed
    torch.save({
        'epoch': end_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'bce_losses': bce_list,
        'kld_losses': kld_list,
        'train_times': train_times + 1,
        'total_training_time': total_training_time,
        'mu_mean': float(np.mean(temp_mu)),
        'mu_std': float(np.mean(temp_std)),
        'logvar_mean': float(np.mean(temp_logvar))
    }, f=LATEST_MODEL_PATH)
    logger.info(f"模型已保存。本次耗时 {elapsed:.2f}s")

def generate_loss_plot():
    """新模式: 生成采样后的 Loss 图表"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型文件。")
        return
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    loss_list = checkpoint.get('train_losses', [])
    bce_list = checkpoint.get('bce_losses', [])
    kld_list = checkpoint.get('kld_losses', [])
    
    if not loss_list: return

    plt.figure(figsize=(15, 5))
    titles = ['Total Loss', 'Recon (BCE+L1)', 'KL Divergence']
    datasets = [loss_list, bce_list, kld_list]
    colors = ['b', 'g', 'r']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        data = datasets[i]
        epochs = np.arange(1, len(data) + 1)
        plt.plot(epochs, data, color=colors[i], alpha=0.3)
        # 采样 20 个点
        num_pts = min(len(data), 20)
        indices = np.linspace(0, len(data) - 1, num_pts, dtype=int)
        plt.scatter(epochs[indices], np.array(data)[indices], color=colors[i], s=30)
        plt.title(titles[i])
        plt.grid(True, linestyle='--', alpha=0.5)

    save_path = MODEL_DIR / f"vae_loss_plot_epoch_{len(loss_list)}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss 图表已保存至: {save_path.name}")

@torch.no_grad()
def evaluate_model():
    """新模式: 评估 VAE 生成质量 (IS + FID + P&R)"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型，请先训练。")
        return

    logger.info("开始评估 VAE 生成质量...")
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    model = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_eval_samples = 500  # 评估样本数
    batch_size = 100
    gen_list = []
    
    start_time = time.time()
    # 1. 从先验分布生成图片
    for i in range(0, total_eval_samples, batch_size):
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        imgs = torch.sigmoid(model.decode_logits(z))
        gen_list.append(imgs)
    all_gen_imgs = torch.cat(gen_list, dim=0)

    # 2. 从测试集获取真实图片
    real_list = []
    collected = 0
    for x, _ in test_loader:
        take = min(x.size(0), total_eval_samples - collected)
        real_list.append(x[:take])
        collected += take
        if collected >= total_eval_samples: break
    all_real_imgs = torch.cat(real_list, dim=0).to(DEVICE)

    try:
        # 3. 计算指标
        logger.info(" -> 计算分类器指标 (置信度/熵/覆盖率)...")
        max_conf, entropy, coverage = evaluate_images(all_gen_imgs, device=DEVICE)

        logger.info(" -> 计算特征流形指标 (FID / Precision / Recall)...")
        fid_score, precision, recall = compute_fid_and_pr(
            gen_images=all_gen_imgs, 
            real_images=all_real_imgs, 
            device=DEVICE
        )

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 55)
        logger.info(f"{'VAE 生成质量评估报告':^50}")
        logger.info("-" * 55)
        logger.info(f" 评估规模     : {total_eval_samples} Samples")
        logger.info(f" 平均置信度   : {max_conf:.4f} (越高越好)")
        logger.info(f" 类别覆盖率   : {coverage:.4f} (越高越好)")
        logger.info(f" FID 分数     : {fid_score:.4f} (越低越好)")
        logger.info(f" Precision    : {precision:.4f} (保真度)")
        logger.info(f" Recall       : {recall:.4f} (多样性)")
        logger.info("=" * 55 + "\n")
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")

# --- 原有功能的辅助函数 ---
def load_model() -> ConvVAE:
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    m = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    return m

def save_large_image(img_data, path: Path, title: str = None, is_compare: bool = False):
    figsize = (8, 4) if is_compare else (4, 4)
    plt.figure(figsize=figsize, dpi=100)
    plt.imshow(img_data, cmap='gray')
    if title: plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def generate_mode():
    current_model = load_model()
    current_timestamp = get_current_time()
    z = torch.randn(COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL, LATENT_DIM, device=DEVICE)
    gen_imgs = torch.sigmoid(current_model.decode_logits(z)).detach().cpu().numpy()
    
    # 保存宫格
    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
        ax.axis('off')
    plt.savefig(IMAGE_DIR / f"vae_gen_{current_timestamp}.png", bbox_inches='tight')
    plt.close()
    logger.info("Generate 模式完成。")

@torch.no_grad()
def compare_mode():
    current_model = load_model()
    loader = DataLoader(test_dataset, batch_size=9, shuffle=True)
    x, _ = next(iter(loader))
    x = x.to(DEVICE)
    mu, logvar = current_model.encode(x)
    z = current_model.reparam(mu, logvar)
    gen = torch.sigmoid(current_model.decode_logits(z)).cpu().numpy()
    orig = x.cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        combined = np.hstack((orig[i][0], gen[i][0]))
        ax.imshow(combined, cmap='gray')
        ax.axis('off')
    plt.savefig(IMAGE_DIR / f"vae_compare_{get_current_time()}.png", bbox_inches='tight')
    plt.close()
    logger.info("Compare 模式完成。")

def show_model_status():
    if not LATEST_MODEL_PATH.exists(): return
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    logger.info(f"\n{'VAE 模型状态':^36}")
    logger.info(f" 训练轮数: {checkpoint.get('epoch')}")
    logger.info(f" Mu 均值:  {checkpoint.get('mu_mean', 0):.4f}")
    logger.info(f" Mu 标准差: {checkpoint.get('mu_std', 0):.4f}")
    logger.info("-" * 36)

# --- 主循环 ---
def main():
    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      VAE 模型管理系统 (Enhanced)")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Compare)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 评估模型生成质量 (Evaluate)")
        logger.info(" [5] 生成损失函数图表 (Loss Plot)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择功能: ").strip().lower()

        if choice == '1':
            train_model()
        elif choice == '2':
            sub_choice = input("  [1] Generate [2] Compare: ").strip()
            if sub_choice == '1': generate_mode()
            elif sub_choice == '2': compare_mode()
        elif choice == '3':
            show_model_status()
        elif choice == '4':
            evaluate_model()
        elif choice == '5':
            generate_loss_plot()
        elif choice in ['0', 'exit']:
            break

if __name__ == '__main__':
    main()