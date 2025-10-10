import os

# 在导入 torch 或 numpy 等库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import create_logger
from pathlib import Path

logger = create_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARENT_DIR = Path(__file__).parent
# 模型的保存位置
model_dir = PROJECT_ROOT / "models"
image_dir = PARENT_DIR / "images"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# 1. 定义 VAE 模型，这里需要搞清楚，h_dim和z_dim分别是什么含义
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        # 为什么这里的 mu 和 logvar 都要用单层神经网络来表示？
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_logvar = nn.Linear(h_dim, z_dim)
        # fc3 和 fc4 在这里起到一个什么作用？
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 从标准正态分布采样
        return mu + eps * std  # 重参数化技巧

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))  # 将图像展平
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 2. 定义损失函数
def vae_loss_function(recon_x, x, mu, logvar):
    # 重构损失 (Binary Cross-Entropy for MNIST images)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')

    # KL 散度损失
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# 3. 设置超参数
INPUT_DIM = 28 * 28  # MNIST 图像大小
H_DIM = 256  # 隐藏层维度
Z_DIM = 20  # 潜在空间维度
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"运行设备：{DEVICE}")

# 4. 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # VAE通常不进行标准化，因为输出是sigmoid，范围是0-1
])

mnist_parent_dir = PROJECT_ROOT / "data"
train_dataset = datasets.MNIST(root=mnist_parent_dir, train=True, transform=transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. 实例化模型、优化器
model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    loss_list = []
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss_list[epoch]:.4f}")
    model_path = model_dir / "mnist_model.pth"
    # torch 在保存模型的时候，应当将模型的参数，运行的次数，以及损失函数都保存进去，而不是仅仅保存参数
    torch.save(model.state_dict(), f=model_path)
    logger.info(f"模型保存成功，位置：{model_path}")
    # 绘制损失函数的图片



def test():
    model.eval()
    with torch.no_grad():
        # 从标准正态分布中采样 Z_DIM 个随机向量
        sample = torch.randn(64, Z_DIM).to(DEVICE)  # 生成64张图像
        generated_images = model.decode(sample).cpu()

        # 可视化生成的图像
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < generated_images.size(0):
                ax.imshow(generated_images[i].view(28, 28), cmap='gray')
                ax.axis('off')
        plt.suptitle("Generated Images")
        plt.show()

        # 可视化重构图像
        print("\nReconstructing images from test set...")
        # 随机取一些测试集图片进行重构
        test_data, _ = next(iter(
            DataLoader(datasets.MNIST(root='../../data', train=False, transform=transform, download=True),
                       batch_size=64)))
        test_data = test_data.to(DEVICE)
        recon_test_images, _, _ = model(test_data)
        recon_test_images = recon_test_images.cpu().view(64, 1, 28, 28)  # 调整形状为 (batch, channel, H, W)

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < test_data.size(0) // 2:
                ax.imshow(test_data[i].view(28, 28).cpu(), cmap='gray')
                ax.axis('off')
            else:
                ax.imshow(recon_test_images[i - test_data.size(0) // 2].view(28, 28), cmap='gray')
                ax.axis('off')
        plt.suptitle("Original (top) vs Reconstructed (bottom) Test Images")
        plt.show()


if __name__ == '__main__':
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train()
    elif mode == "test":
        test()
    else:
        print("Please enter mode ('train' or 'test')")
