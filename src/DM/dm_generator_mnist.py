from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from utils.logger import create_logger

logger = create_logger(__name__)
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "dm_mnist_latent.pth"


# --- 1. 配置参数 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 32  # MNIST 尺寸
batch_size = 64
lr = 1e-4
num_epochs = 30  # 扩散模型训练较慢，建议 30-50 epoch 效果极佳

# --- 2. 数据准备 ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # 映射到 [-1, 1]
])
dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 定义模型 (使用 diffusers 的标准 UNet2D) ---
# 这个模型自带了 ResNet 块、注意力机制和时间嵌入，效果远超简单卷积
model = UNet2DModel(
    sample_size=image_size,  # 输入图片的分辨率
    in_channels=1,  # 输入通道
    out_channels=1,  # 输出通道
    layers_per_block=2,  # 每个 Block 的层数
    block_out_channels=(64, 128, 128, 256),  # 每一层的通道数
    down_block_types=(
        "DownBlock2D",  # 正常的卷积下采样
        "DownBlock2D",
        "AttnDownBlock2D",  # 带 Self-Attention 的下采样 (效果关键)
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",  # 带 Self-Attention 的上采样
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# --- 4. 定义噪声调度器 (Scheduler) ---
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- 5. 训练循环 ---
for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for step, (clean_images, _) in enumerate(progress_bar):
        clean_images = clean_images.to(device)

        # 为每张图片随机生成噪声
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # 随机选择时间步 t
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        # 前向过程：根据 t 将噪声加到图片上
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # 模型预测噪声
        noise_pred = model(noisy_images, timesteps).sample

        # 计算损失 (预测噪声与实际噪声的 MSE)
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

    # --- 6. 生成并保存图片 (每个 Epoch 结束后看效果) ---
    if (epoch + 1) % 5 == 0 or epoch == 0:
        # 使用 diffusers 封装好的 Pipeline 进行推理
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        images = pipeline(batch_size=16, num_inference_steps=1000).images

        # 保存图片
        for i, img in enumerate(images):
            if not os.path.exists(IMAGE_DIR / f"epoch_{epoch}"):
                os.makedirs(IMAGE_DIR / f"epoch_{epoch}", exist_ok=True)
            img.save(IMAGE_DIR / f"epoch_{epoch}/sample_{i}.png")

torch.save(model.state_dict(), LATEST_MODEL_PATH)

print("训练完成！")