import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from utils.logger import create_logger

logger = create_logger(__name__)
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
MNIST_DIR = PROJECT_ROOT_DIR / 'data' / 'MNIST'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "unet_mnist_latent.pth"

# --- 1. 结构完全对齐之前的 ConvVAE ---
# from run_vae import ConvVAE
class ConvVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )
        self.fc_mu = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.decoder_input = nn.Conv2d(latent_channels, 64, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),  # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)


# --- 2. 配置参数 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_channels = 4
latent_size = 4  # 潜空间分辨率 4x4
batch_size = 128
epochs = 100
lr = 1e-4

# 数据准备
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 初始化模型 ---
# 加载预训练的 VAE (假设你已经保存了权重)
vae = ConvVAE(latent_channels=latent_channels).to(device)
if os.path.exists(MODEL_DIR /  "latest_model.pth"):
    # 注意：这里加载你之前的 VAE 权重
    checkpoint = torch.load(MODEL_DIR /  "latest_model.pth", map_location=device)
    # 兼容性处理：如果保存的是整个字典，取 model_state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    vae.load_state_dict(state_dict)
    logger.info("VAE 权重加载成功")
else:
    logger.error("VAE 模型不存在")
vae.eval()

# 定义扩散模型的 U-Net (针对 4x4 潜空间)
# diffusers 的 UNet2DModel 非常灵活
unet = UNet2DModel(
    sample_size=latent_size,  # 潜空间分辨率
    in_channels=latent_channels,
    out_channels=latent_channels,
    layers_per_block=2,
    block_out_channels=(64, 128),  # 因为是 4x4，不需要太多下采样
    down_block_types=("DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D"),
).to(device)

# 定义调度器 (DDPM)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

# --- 4. 训练 LDM ---

def train_ldm():
    loss_list = []
    start_epoch = 0
    train_times = 0
    # 这里需要做一个判断，判断是否有加载的模型
    if os.path.exists(LATEST_MODEL_PATH):
        # 获取上一次训练的检查点
        check_point = torch.load(LATEST_MODEL_PATH, map_location=device)
        logger.info(f"已存在训练的模型, epoch={check_point['epoch']}")
        # 加载上一次训练的模型的权重
        unet.load_state_dict(check_point["model_state_dict"])
        # 加载上一次训练模型的优化器
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        loss_list = check_point["train_losses"]
        start_epoch = check_point["epoch"]
        train_times = check_point["train_times"]
    else:
        logger.info(f"第一次训练模型")

    logger.info("开始训练 Latent Diffusion Model...")
    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"Epoch {epoch + 1}/{start_epoch + epochs}")
        avg_loss = 0

        for step, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)):
            clean_images = batch.to(device)

            # 1. 使用 VAE 获得潜变量 (Latents)
            with torch.no_grad():
                latents, _ = vae.encode(clean_images)
                # 缩放 latent 以保持方差近似为 1（扩散模型训练的 trick）
                latents = latents * 0.18215

                # 2. 采样噪声并加噪
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. 预测噪声
            noise_pred = unet(noisy_latents, timesteps).sample

            # 4. 计算损失
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
        loss_list.append(avg_loss)

    # torch 在保存模型的时候，应当将模型的参数，运行的次数，以及损失函数都保存进去，而不是仅仅保存参数
    torch.save(unet.state_dict(), f=LATEST_MODEL_PATH)
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1
    }, f=LATEST_MODEL_PATH)
    logger.info(f"模型保存成功，位置：{LATEST_MODEL_PATH}")
    # 绘制损失函数的图片
    plt.plot(loss_list)
    plt.savefig(MODEL_DIR / "ldm_loss.png")


# --- 5. 推理生成 (LDM Inference) ---
def generate_samples(num_samples=16):
    unet.eval()
    # 1. 从纯噪声开始 [num, 4, 4, 4]
    sample = torch.randn(num_samples, latent_channels, latent_size, latent_size).to(device)

    # 2. 逐步去噪循环
    for t in tqdm(noise_scheduler.timesteps):
        # 预测噪声残差
        with torch.no_grad():
            residual = unet(sample, t).sample

        # 计算上一步的样本 (x_t -> x_{t-1})
        sample = noise_scheduler.step(residual, t, sample).prev_sample

    # 3. 反缩放并使用 VAE 解码
    sample = sample / 0.18215
    with torch.no_grad():
        images = vae.decode(sample)

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(IMAGE_DIR / f"ldm_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == '__main__':
    train_ldm()
    logger.info("训练完成，开始采样图片...")
    generate_samples()