import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opentelemetry.context import detach
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"

# --- 1. 参数设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_channels = 4
latent_spatial_size = 4
NUM_EPOCHS = 500
lr = 1e-3

# --- 2. 加载 MNIST 数据集 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 3. 定义卷积 VAE 模型 ---
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # 编码器：(1, 28, 28) -> (latent_channels, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 4, 4)
            nn.ReLU(),
        )

        # 用于计算 mu 和 logvar 的卷积层（保持空间结构）
        # 我们用卷积层代替全连接层，输出依然是 (C, 4, 4)
        self.fc_mu = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(64, latent_channels, kernel_size=1)

        # 解码器：从 (latent_channels, 4, 4) 还原到 (1, 28, 28)
        self.decoder_input = nn.Conv2d(latent_channels, 64, kernel_size=1)

        self.decoder = nn.Sequential(
            # output_padding 用于精确匹配尺寸：4->7->14->28
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # -> (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- 4. 损失函数 ---
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    logger.info(f"损失记录：BCE={BCE.item()}, KLD={KLD.item()}, 合计损失：{BCE.item() + KLD.item()}")
    return BCE + 2 * KLD


# --- 5. 训练模型 ---
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()


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

    # 开始执行训练
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{start_epoch + NUM_EPOCHS}")
        avg_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
        logger.info(f"Epoch [{epoch + 1}/{start_epoch + NUM_EPOCHS}], avg loss: {avg_loss:.2f}")
        loss_list.append(avg_loss)

    # torch 在保存模型的时候，应当将模型的参数，运行的次数，以及损失函数都保存进去，而不是仅仅保存参数
    torch.save(model.state_dict(), f=LATEST_MODEL_PATH)
    torch.save({
        'epoch': start_epoch + NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'train_times': train_times + 1
    }, f=LATEST_MODEL_PATH)
    logger.info(f"模型保存成功，位置：{LATEST_MODEL_PATH}")
    # 绘制损失函数的图片
    fig = plt.figure()
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_title("avg loss curve")
    axes1.plot(loss_list)
    print(f"loss_list: {len(loss_list)}")
    # plt.show()
    plt.savefig(MODEL_DIR / "avg_loss.png")


def test_model():
    # 获取上一次训练的检查点
    check_point = torch.load(LATEST_MODEL_PATH, map_location=device)
    current_epoch = check_point["epoch"]
    logger.info(f"已存在训练的模型, epoch={check_point['epoch']}")
    # 加载上一次训练的模型的权重
    model.load_state_dict(check_point["model_state_dict"])
    model.eval()

    with torch.no_grad():
        while True:
            user_input = input("请选择要测试的模式 (1) generate (2) compare (输入 exit 或 0 退出)：").strip().lower()
            if user_input in ["exit", "0"]:
                logger.info("退出测试模式")
                break
            elif user_input in ["1", "generate"]:
                logger.info("执行 generate 模式")
                # 从标准正态分布中采样
                sample = torch.randn(8, latent_channels, latent_spatial_size, latent_spatial_size).to(device)
                generated_images = model.decode(sample).cpu()

                # 可视化生成的图像
                fig, axes = plt.subplots(2, 4, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    if i < generated_images.size(0):
                        ax.imshow(generated_images[i].view(28, 28), cmap='gray')
                        ax.axis('off')
                plt.suptitle(f"Generated Images epoch: {current_epoch}")
                save_path = IMAGE_DIR / f"generated_image_{current_epoch}.png"
                plt.savefig(save_path)
                logger.info(f"图片生成完毕，已保存至 {save_path}")
            elif user_input in ["2", "compare"]:
                logger.info("执行 compare 模式")
                logger.info("\nReconstructing images from test set...")
                # 随机取一个测试 batch
                test_data, _ = next(iter(
                    DataLoader(datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=False),
                               batch_size=64)))
                test_data = test_data.to(device)
                recon_test_images, _, _ = model.forward(test_data)
                recon_test_images = recon_test_images.cpu().view(64, 1, 28, 28)

                # 绘制对比图（前32张为原图，后32张为重构）
                fig, axes = plt.subplots(8, 8, figsize=(8, 8))
                for i, ax in enumerate(axes.flat):
                    if i < test_data.size(0) // 2:
                        ax.imshow(test_data[i].view(28, 28).cpu(), cmap='gray')
                        ax.axis('off')
                    else:
                        ax.imshow(recon_test_images[i - test_data.size(0) // 2].view(28, 28), cmap='gray')
                        ax.axis('off')
                plt.suptitle(f"Compare Image epoch: {current_epoch}")
                save_path = IMAGE_DIR / f"compare_image_{current_epoch}.png"
                plt.savefig(save_path)
                logger.info(f"图像对比完成，已保存至 {save_path}")
            else:
                logger.error("输入错误，请重新输入")
                continue

if __name__ == '__main__':
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    else:
        logger.error("Please enter mode ('train' or 'test')")
