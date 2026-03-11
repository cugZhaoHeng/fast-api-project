import os
import torch
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

# --- 1. 参数设置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 28  # MNIST 的分辨率
batch_size = 128
epochs = 45  # 建议训练 50 轮以上效果更佳
lr = 1e-4

# --- 2. 数据准备 ---
# 扩散模型通常喜欢将数据归一化到 [-1, 1] 之间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 初始化组件 ---

# 定义 U-Net 模型
# 这里的输入和输出都是 1 个通道 (黑白图)
model = UNet2DModel(
    sample_size=image_size,  # 输入图像的尺寸
    in_channels=1,  # 输入通道
    out_channels=1,  # 输出通道
    layers_per_block=2,
    block_out_channels=(64, 128, 128),  # 每一层的特征通道数
    down_block_types=(
        "DownBlock2D",  # 普通下采样
        "AttnDownBlock2D",  # 带注意力机制的下采样
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",  # 普通上采样
    ),
).to(device)

# 定义调度器 (负责加噪逻辑)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- 4. 训练循环 ---
def train_model():
    loss_list = []
    start_epoch = 0
    train_times = 0
    # 这里需要做一个判断，判断是否有加载的模型
    if os.path.exists(LATEST_MODEL_PATH):
        # 获取上一次训练的检查点
        check_point = torch.load(LATEST_MODEL_PATH, map_location=device)
        logger.info(f"已存在训练的模型, epoch={check_point['epoch']}")
        # 加载上一次训练的模型的权重
        model.load_state_dict(check_point["model_state_dict"])
        # 加载上一次训练模型的优化器
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        loss_list = check_point["train_losses"]
        start_epoch = check_point["epoch"]
        train_times = check_point["train_times"]
    else:
        logger.info(f"第一次训练模型")

    logger.info(f"开始在像素空间训练扩散模型 (Device: {device})...")
    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"Epoch {epoch + 1}/{start_epoch + epochs}")
        avg_loss = 0

        for step, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)):
            clean_images = batch.to(device)

            # 1. 为图片采样噪声
            noise = torch.randn(clean_images.shape).to(device)

            # 2. 随机采样时间步 t (0 到 999)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],), device=device
            ).long()

            # 3. 前向加噪：根据 t 将噪声加到图片上
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # 4. 预测：让模型预测加进去的噪声是多少
            noise_pred = model(noisy_images, timesteps).sample

            # 5. 计算损失 (预测噪声 vs 实际噪声)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
        loss_list.append(avg_loss)

    # torch 在保存模型的时候，应当将模型的参数，运行的次数，以及损失函数都保存进去，而不是仅仅保存参数
    torch.save(model.state_dict(), f=LATEST_MODEL_PATH)
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1
    }, f=LATEST_MODEL_PATH)
    logger.info(f"模型保存成功，位置：{LATEST_MODEL_PATH}")
    # 绘制损失函数的图片
    plt.plot(loss_list)
    plt.savefig(MODEL_DIR / "ldm_loss.png")


# --- 5. 生成与推理 (Sampling) ---
def generate_and_show(num_samples=16):
    model.eval()
    # 1. 从纯高斯噪声开始 [num, 1, 28, 28]
    sample = torch.randn(num_samples, 1, image_size, image_size).to(device)

    # 2. 逐步去噪
    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            # 预测噪声
            residual = model(sample, t).sample

        # 根据模型预测的噪声，计算上一个时间步的图像 (x_t -> x_{t-1})
        sample = noise_scheduler.step(residual, t, sample).prev_sample

    # 3. 结果转换：从 [-1, 1] 变回 [0, 1] 用于显示
    sample = (sample / 2 + 0.5).clamp(0, 1)

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle("Generated Images (Pixel Space)")
    plt.show()

if __name__ == '__main__':
    train_model()
    logger.info("训练结束，开始采样图片...")
    generate_and_show()