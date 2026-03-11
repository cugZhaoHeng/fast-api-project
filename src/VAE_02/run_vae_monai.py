import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from generative.networks.nets import AutoencoderKL
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import numpy as np

# 假设你的 logger 已经配置好，如果没有请注释掉相关行
from utils.logger import create_logger
logger = create_logger(__name__)

# --- 1. 参数设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_channels = 1  # 按你之前的设定改为 1
latent_spatial_size = 4  # 32 / 8 = 4
NUM_EPOCHS = 10  # 建议训练多几轮，MONAI 模型较深
lr = 1e-4  # 学习率建议稍微调小一点，利于收敛
kl_weight = 1e-6  # 论文中推荐的 KL 权重，防止 KLD 占用过高导致图片模糊

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "monai_vae_mnist.pth"

# --- 2. 加载 MNIST 数据集 (包含 Resize 预处理) ---
# 将 28x28 放大到 32x32，以适配 3 层下采样 (32 -> 16 -> 8 -> 4)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 3. 定义模型 ---
model = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    latent_channels=latent_channels,
    num_res_blocks=2,
    norm_num_groups=32,
    attention_levels=(False, False, True),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)


# --- 4. 修改后的损失函数 (适配 z_sigma) ---
def loss_function(recon_x, x, mu, sigma):
    # 1. 重构损失 (L1 损失在地质建模中常用，比 MSE 边缘更清晰)
    recons_loss = F.l1_loss(recon_x, x, reduction='sum')

    # 2. KL 散度 (注意：这里使用 sigma 而不是 logvar)
    # 公式: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    kl_loss = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1)

    return recons_loss, kl_loss


# --- 5. 训练逻辑 ---
def train_model():
    start_epoch = 0
    if os.path.exists(LATEST_MODEL_PATH):
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"已加载模型，继续从 Epoch {start_epoch} 开始训练")

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # MONAI VAE 返回的是: reconstruction, mu, sigma
            recon_batch, mu, sigma = model(data)

            recons_l, kl_l = loss_function(recon_batch, data, mu, sigma)
            loss = recons_l + kl_weight * kl_l

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item() / len(data):.4f} (Recon: {recons_l.item() / len(data):.4f}, KL: {kl_l.item() / len(data):.4f})")

        # 保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, LATEST_MODEL_PATH)


# --- 6. 测试与可视化 ---
def test_model():
    if not os.path.exists(LATEST_MODEL_PATH):
        print("未找到模型文件")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        mode = input("请选择模式 (1) generate (2) compare: ")

        if mode == "1" or mode == "generate":
            # 随机生成潜变量 [Batch, Latent_C, 4, 4]
            sample = torch.randn(16, latent_channels, latent_spatial_size, latent_spatial_size).to(device)
            generated = model.decode(sample).cpu()

            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated[i].squeeze(), cmap='gray')
                ax.axis('off')
            plt.suptitle("Generated (MONAI AutoencoderKL)")
            plt.savefig(IMAGE_DIR / f"vae_monai_image_generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        elif mode == "2" or mode == "compare":
            # 获取一批测试数据 (记得也要 Resize)
            dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform)
            test_loader = DataLoader(dataset, batch_size=8, sampler=RandomSampler(dataset, replacement=True, num_samples=8))
            data, _ = next(iter(test_loader))
            data = data.to(device)
            recon, _, _ = model(data)

            fig, axes = plt.subplots(2, 8, figsize=(12, 4))
            for i in range(8):
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')
                axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title("Recon")
                axes[1, i].axis('off')
            plt.savefig(IMAGE_DIR / f"vae_monai_image_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    logger.info(f"图片已保存至文件夹 {IMAGE_DIR}")


if __name__ == '__main__':
    action = input("Enter 'train' or 'test': ").strip().lower()
    if action == 'train':
        train_model()
    elif action == 'test':
        test_model()