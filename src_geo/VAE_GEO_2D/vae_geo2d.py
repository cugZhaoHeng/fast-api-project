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

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
GEO_IMAGE_DIR = DATA_DIR / 'geo_images'

IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"
COMBINE_IMAGES_ROW: int = 3  # 合并后的图像的行数，用来形成一副多个子图的整体图片
COMBINE_IMAGES_COL: int = 3  # 合并后的图像的列数，用来形成一副多个子图的整体图片
IMAGE_WIDTH: int = 64  # 图片的宽度像素
IMAGE_HEIGHT: int = 64  # 图片的高度像素

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time
from utils.image_dataloader import get_vae_dataloaders

logger = create_logger(__name__)

# --- 1. 参数设置 ---
DEVICE = init_gpu_environment()
BATCH_SIZE = 128
NUM_EPOCHS = 1000

LR = 1e-3
WEIGHT_DECAY = 1e-5
LATENT_DIM = 24

# --- 2. 加载 MNIST 数据集 ---
# 设置 transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset, train_loader, test_loader = get_vae_dataloaders(GEO_IMAGE_DIR, transform)
logger.info(f"train_loader: {len(train_loader)}")
logger.info(f"test_loader: {len(test_loader)}")

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
    def __init__(self, z_dim=128):
        super().__init__()
        self.z_dim = z_dim

        # Encoder: 1x64x64 -> 256x4x4
        # 经历 4 次下采样 (Stride=2): 64->32, 32->16, 16->8, 8->4
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),      # 32x32
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),

            nn.Conv2d(32, 64, 4, 2, 1),     # 16x16
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResBlock(64),

            nn.Conv2d(64, 128, 4, 2, 1),    # 8x8
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ResBlock(128),

            nn.Conv2d(128, 256, 4, 2, 1),   # 4x4
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        # 4x4 * 256 = 4096
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

        # Decoder: z -> 1x64x64
        self.fc_dec = nn.Linear(z_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8x8
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1)     # 64x64 (Logits)
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
        # 关键修正：从 z 还原回 4x4, 256通道 的特征图
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

def beta_schedule(epoch):
    if epoch >= 40:
        return 0.15
    return 0.15 * (epoch / max(1, 40))


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
def train_model():
    model = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    loss_list, bce_list, kld_list = [], [], []  # 损失函数记录
    mu_mean, mu_std, logvar_mean = 0, 0, 0
    start_epoch = 0
    
    train_times = 0  # 训练次数，也就是执行了多少次 train_model 函数
    total_training_time = 0.0  # 累计训练总时长（秒）

    # 记录本次运行开始时间
    start_time = time.time()

    # 如果本地存在预训练的模型，则直接加载
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
        logger.info(
            f"已加载模型, 起始 epoch={start_epoch}， 终止 epoch={start_epoch + NUM_EPOCHS}, 累计训练时长={total_training_time:.2f}s")
    else:
        logger.info("第一次训练模型")
    end_epoch = start_epoch + NUM_EPOCHS

    for epoch in range(start_epoch, end_epoch):
        avg_loss, avg_bce, avg_kld = 0, 0, 0
        temp_mu, temp_std, temp_logvar = [], [], []

        beta = beta_schedule(epoch)
        total_loss, total_n = 0.0, 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            bs = x.size(0)  # 注意，bs不一定就是代码首部定义的 BATCH_SIZE，因为最后一次遍历的数量会小于 BATCH_SIZE

            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model.forward(x)
            loss, bce, kld = loss_fn(
                logits, x, mu, logvar,
                beta=beta,
                free_bits_per_dim=0.02,
                l1_weight=0.15
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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
        
        mu_mean = float(np.mean(temp_mu))
        mu_std = float(np.mean(temp_std))
        logvar_mean = float(np.mean(temp_logvar))

        logger.info(
            f"Epoch [{epoch + 1}/{end_epoch}] Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f}), beta: {beta}")

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
        'total_training_time': total_training_time,
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'logvar_mean': logvar_mean
    }, f=LATEST_MODEL_PATH)

    logger.info(f"模型已保存，本次训练耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s")

    plt.figure(figsize=(15, 5))
    titles = ['Total Loss', 'BCE (Recon Loss)', 'KLD (KL Loss)']
    data_to_plot = [loss_list, bce_list, kld_list]
    colors = ['b', 'g', 'r']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        curr_data = data_to_plot[i]
        epochs_range = np.arange(1, len(curr_data) + 1)
        plt.plot(epochs_range, curr_data, color=colors[i], alpha=0.3)

        # 20点采样逻辑
        num_pts = min(len(curr_data), 20)
        indices = np.linspace(0, len(curr_data) - 1, num_pts, dtype=int)
        plt.scatter(epochs_range[indices], np.array(curr_data)[indices], color=colors[i], s=30)

        plt.title(titles[i])
        plt.xlabel("epochs")
        plt.grid(True, linestyle='--', alpha=0.5)

    current_timestamp: str = get_current_time()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"loss_{current_timestamp}_epoch_{end_epoch}.png")
    plt.close()
    logger.info("指标图表已保存。")


def show_model_status():
    """查看当前模型参数及隐空间分布状态"""
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
        logger.info("=" * 40 + "\n")

    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")


def save_large_image(img_data, path: Path, title: str = None, is_compare: bool = False):
    """保存大尺寸图片的辅助函数"""
    # 如果是对比图，宽度加倍
    figsize = (8, 4) if is_compare else (4, 4)
    plt.figure(figsize=figsize, dpi=100)  # 100 DPI 下 4英寸=400像素

    if is_compare:
        plt.imshow(img_data, cmap='gray')
    else:
        plt.imshow(img_data, cmap='gray')

    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# 从本地加载 pth 格式的模型文件，并使用 evaluate 模式
def load_model() -> ConvVAE:
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
    current_model = ConvVAE(z_dim=LATENT_DIM).to(DEVICE)
    current_model.load_state_dict(checkpoint['model_state_dict'])
    current_model.eval()
    return current_model


@torch.no_grad()
def generate_mode():
    """模式 2-1: 生成 9 张大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    current_model = load_model()
    current_timestamp: str = get_current_time()
    z = torch.randn(COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL, LATENT_DIM, device=DEVICE)
    # 保存前，将 pytorch的张量转化为 numpy 的数组形式
    gen_imgs = torch.sigmoid(current_model.decode_logits(z)).detach().cpu().numpy()

    # 1. 保存 9 张独立大图
    for i in range(COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL):
        # 命名: vae_image_generate_时间戳_编号.png
        filename = IMAGE_DIR / f"vae_image_generate_{current_timestamp}_{i + 1:02d}.png"
        # 将图片写入到本地 images 文件夹
        save_large_image(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), filename)

    # 2. 保存 3*3 宫格图
    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
        ax.axis('off')
    grid_fn = IMAGE_DIR / f"vae_image_generate_{current_timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("Generate 模式运行完毕，图片已保存。")


@torch.no_grad()
def compare_mode():
    """模式 2-2: 重构 9 张对比大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    current_model = load_model()
    train_dataset, train_loader, test_loader = get_vae_dataloaders(GEO_IMAGE_DIR, transform, batch_size=COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL, split_ratio=1.0)

    # 收集原始图像和对应的生成图像
    orig_list = []  # 存放原始图像张量
    gen_list = []  # 存放生成图像张量
    need = 9
    it = iter(train_loader)

    while need > 0:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, _ = next(it)
        x = x.to(DEVICE, non_blocking=True)
        take = min(need, x.size(0))
        orig_list.append(x[:take])

        # 对这批图像编码后解码（使用聚合后验采样）
        mu, logvar = current_model.encode(x[:take])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        gen = torch.sigmoid(current_model.decode_logits(z)).cpu()
        gen_list.append(gen)

        need -= take

    # 合并所有批次
    orig = torch.cat(orig_list, dim=0).cpu()
    gen = torch.cat(gen_list, dim=0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 转为 numpy 并去除通道维度
    orig_np = orig.squeeze().numpy()
    recon_np = gen.squeeze().numpy()

    # 1. 保存 9 张独立对比大图
    for i in range(COMBINE_IMAGES_ROW * COMBINE_IMAGES_COL):
        # 左右拼接
        combined = np.hstack((orig_np[i], recon_np[i]))
        filename = IMAGE_DIR / f"vae_image_compare_{timestamp}_{i + 1:02d}.png"
        save_large_image(combined, filename, title="Original | Reconstructed", is_compare=True)

    # 2. 保存 3*3 宫格对比总图
    fig, axes = plt.subplots(COMBINE_IMAGES_ROW, COMBINE_IMAGES_COL, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for i, ax in enumerate(axes.flat):
        combined = np.hstack((orig_np[i], recon_np[i]))
        ax.imshow(combined, cmap='gray')
        ax.set_title(f"Pair {i + 1:02d}")
        ax.axis('off')
    grid_fn = IMAGE_DIR / f"vae_image_compare_{timestamp}.png"
    plt.savefig(grid_fn, bbox_inches='tight')
    plt.close()
    logger.info("Compare 模式运行完毕，图片已保存。")


def main():
    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      VAE 模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Compare)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 评估模型 (Evaluate)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

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
                elif sub_choice in ['0', 'exit']:
                    logger.info("退出程序...")
                    sys.exit(0)
                else:
                    logger.info("  无效输入，请输入 1, 2 或 0")
        elif choice == '3':
            show_model_status()
        elif choice in ['0', 'exit']:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3 或 0")


if __name__ == '__main__':
    main()
