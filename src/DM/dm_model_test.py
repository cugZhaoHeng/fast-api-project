from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
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
LATEST_MODEL_PATH = MODEL_DIR / "dm_mnist_latent.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 32


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

checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
checkpoint_keys = checkpoint.keys()
print(checkpoint_keys)
model.load_state_dict(checkpoint)
model.eval()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
images = pipeline(batch_size=16, num_inference_steps=1000).images

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

plt.tight_layout()
plt.show()