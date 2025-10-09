import logging
import os
from pathlib import Path
from tqdm import tqdm

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
from utils import logger
log = logger.create_logger(__name__)
log.setLevel(logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

# 1. 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_logvar = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        log.debug(f"输入x: {x.shape}")
        # 1. 执行一次全连接输出
        x = self.fc1(x)
        log.debug(f"x: {x.shape}")
        h = self.relu(x)
        mu = self.fc2_mu(h)
        log.debug(f"mu: {mu.shape}")
        logvar = self.fc2_logvar(h)
        log.debug(f"logvar: {logvar.shape}")
        log.debug(f"mu: {mu}")
        log.debug(f"logvar: {logvar}")
        return mu, logvar

    def decode(self, z):
        log.debug(f"z: {z.shape}")
        z = self.fc3(z)
        log.debug(f"z: {z.shape}")
        h = self.relu(z)
        h = self.fc4(h)
        log.debug(f"h: {h.shape}")
        h = self.sigmoid(h)
        log.debug(f"h: {h.shape}")
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 从标准正态分布采样
        return mu + eps * std # 重参数化技巧

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1)) # 将图像展平
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

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
H_DIM = 256         # 隐藏层维度
Z_DIM = 20          # 潜在空间维度
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # VAE通常不进行标准化，因为输出是sigmoid，范围是0-1
])

train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. 实例化模型、优化器
model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model_filename = "model3.pth"

def train():
    # 6. 训练模型
    print(f"Training on {DEVICE}")
    for epoch in range(NUM_EPOCHS):
        for data, _ in tqdm(train_loader):
            data = data.to(DEVICE)

            recon_batch, mu, logvar = model.forward(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_filename)
    log.info(f"训练完成，模型保存至 {model_filename}")
def test():
    with open(file='model3.pth', mode='rb') as f:
        new_model = torch.load(f=f, map_location=DEVICE, weights_only=True)
    model1 = VAE(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM)
    model1.load_state_dict(state_dict=new_model)
    model1.to(DEVICE)
    log.info(f"model1: {model1}")
    # 7. 生成新图像
    print("\nGenerating new images...")
    model1.eval()
    with torch.no_grad():
        # 从标准正态分布中采样 Z_DIM 个随机向量
        sample = torch.randn(64, Z_DIM).to(DEVICE) # 生成64张图像
        log.info(f"sample[0] shape: {sample[0].shape}")
        log.info(f"sample[0]: {sample[0]}")
        generated_images = model1.decode(sample)
        generated_images = generated_images.cpu().numpy()

        log.info(f"generated_images[0] shape: {generated_images[0].shape}")
        log.info(f"generated_images[0]: {generated_images[0].reshape(28,28)}")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for i in range(2):
            for j in range(2):
                axes[i, j].imshow(generated_images[i * 2 + j].reshape(28, 28), cmap='gray')
        plt.show()
        plt.close()


        # # 可视化重构图像
        print("\nReconstructing images from test set...")
        # 随机取一些测试集图片进行重构
        test_data, _ = next(iter(DataLoader(datasets.MNIST(root='../../data', train=False, transform=transform, download=True), batch_size=64)))
        # 绘制 test_data成图片
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
        for i in range(2):
            for j in range(4):
                axes[i, j].imshow(test_data[i * 2 + j].reshape(28, 28), cmap='gray')
        plt.show()
        plt.close()


        test_data = test_data.to(DEVICE)
        recon_test_images, _, _ = model1(test_data)
        log.info(f"recon_test_images: {recon_test_images.shape}")
        recon_test_images = recon_test_images = recon_test_images.cpu().numpy().reshape(-1, 28, 28)
        log.info(f"recon_test_images: {recon_test_images.shape}")

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
        for i in range(2):
            for j in range(4):
                axes[i, j].imshow(recon_test_images[i * 2 + j].reshape(28, 28), cmap='gray')
        plt.show()
        plt.close()

if __name__ == '__main__':
    train()
    # test()