import torch
from torchvision.utils import save_image
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
G_LATEST_MODEL_PATH = MODEL_DIR / "g_latest_model.pth"
D_LATEST_MODEL_PATH = MODEL_DIR / "d_latest_model.pth"

# --- 1. 超参数设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
lr = 0.0002
latent_size = 100  # 噪声向量的长度
image_size = 28 * 28  # MNIST 是 28x28
epochs = 100

# --- 2. 数据加载 ---
# 将图片归一化到 [-1, 1]，这对 GAN 的稳定性很重要
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mnist = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=False)
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
# --- 3. 逐个保存前 8 个样本 ---

# 获取一个批次的数据
data_iter = iter(data_loader)
images, labels = next(data_iter)

print("正在生成高清图片...")

for i in range(8):
    img_tensor = images[i]  # 获取 Tensor (1, 28, 28)
    label = labels[i].item()  # 获取标签数字

    # --- 关键步骤：数据预处理 ---
    # 1. 去归一化：将 [-1, 1] 还原回 [0, 1] 以便 matplotlib 显示
    # 公式：image = image * std + mean
    img_denorm = img_tensor * 0.5 + 0.5

    # 2. 转换格式：Tensor (C, H, W) -> Numpy (H, W)
    # .squeeze() 去掉通道维度，变成 (28, 28)
    img_numpy = img_denorm.squeeze().numpy()

    # --- 调用您的保存函数 ---
    title = f"Label: {label}"
    # 保存路径，例如: images/sample_0_label_5.png
    save_path = IMAGE_DIR / f"sample_{i}_label_{label}.png"

    save_large_image(img_numpy, save_path, title=title, is_compare=False)

    print(f"已保存: {save_path}")

print("✅ 所有高清图片已生成！")