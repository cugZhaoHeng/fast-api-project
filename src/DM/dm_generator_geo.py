from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from PIL import Image
import os
import glob
from tqdm import tqdm

from utils.logger import create_logger

logger = create_logger(__name__)
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
GEO_IMAGE_DIR = PROJECT_ROOT_DIR / 'data' / 'geo_images'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "unet_mnist_latent.pth"

# --- 1. 自定义地质数据集 (与 GAN 版本一致) ---
class GeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 注意：地质模型如果是彩色则用'RGB'，单色用'L'
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# --- 2. 超参数设置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 64
batch_size = 32  # 64x64 较占显存，若显存够可改 64
lr = 1e-4
num_epochs = 50  # 扩散模型收敛慢，地质模型建议 50-100 epoch
nc = 3  # 通道数

# --- 3. 数据加载 ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),  # 地质模型增加水平翻转增强多样性
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # 映射到 [-1, 1]
])

dataset = GeoDataset(root_dir=GEO_IMAGE_DIR, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 4. 定义针对 64x64 优化的 U-Net ---
model = UNet2DModel(
    sample_size=image_size,
    in_channels=nc,
    out_channels=nc,
    layers_per_block=2,
    # 针对 64x64，我们增加一层通道配置 (64 -> 32 -> 16 -> 8 -> 4)
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",  # 64x64 -> 32x32
        "DownBlock2D",  # 32x32 -> 16x16
        "DownBlock2D",  # 16x16 -> 8x8
        "AttnDownBlock2D",  # 8x8 -> 4x4 (在地质结构深层加入注意力机制)
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# --- 5. 调度器与优化器 ---
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- 6. 训练循环 ---
os.makedirs("geo_diffusion_results", exist_ok=True)

for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for step, clean_images in enumerate(progress_bar):
        clean_images = clean_images.to(device)
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # 采样随机时间步
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        # 向图像添加噪声 (Forward Diffusion)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # 预测噪声 (Backward Diffusion)
        noise_pred = model(noisy_images, timesteps).sample

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

    # --- 7. 每 10 个 Epoch 生成一次地质图预览 ---
    if (epoch + 1) % 10 == 0:
        model.eval()
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        # 生成 16 张地质图
        images = pipeline(batch_size=16, num_inference_steps=1000).images

        # 拼接并保存
        grid_img = Image.new('RGB', (image_size * 4, image_size * 4))
        for i, img in enumerate(images):
            grid_img.paste(img, ((i % 4) * image_size, (i // 4) * image_size))
        grid_img.save(f"geo_diffusion_results/epoch_{epoch + 1}.png")
        model.train()

# 保存最终模型权重
torch.save(model.state_dict(), "geo_diffusion_model.pth")