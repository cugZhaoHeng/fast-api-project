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


# --- 3. 定义网络结构 ---

# 判别器：本质是一个二分类器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 0~1 之间的概率
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 拉平图片
        return self.model(x)


# 生成器：将噪声变为图片
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size),
            nn.Tanh()  # 将输出限制在 [-1, 1] 以匹配数据预处理
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)


# 实例化
D = Discriminator().to(device)
G = Generator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵
d_optimizer = optim.Adam(D.parameters(), lr=lr)
g_optimizer = optim.Adam(G.parameters(), lr=lr)

# --- 4. 训练循环 ---
if not os.path.exists('samples'): os.makedirs('samples')

def train_model():
    g_loss_list = []
    d_loss_list = []
    start_epoch = 0
    train_times = 0

    # 这里需要做一个判断，判断是否有加载的模型
    if os.path.exists(G_LATEST_MODEL_PATH):
        # 获取上一次训练的检查点
        check_point = torch.load(G_LATEST_MODEL_PATH, map_location=device)
        logger.info(f"已存在训练的模型, epoch={check_point['epoch']}")
        # 加载上一次训练的模型的权重
        G.load_state_dict(check_point["model_state_dict"])
        # 加载上一次训练模型的优化器
        g_optimizer.load_state_dict(check_point["optimizer_state_dict"])
        g_loss_list = check_point["train_losses"]
        start_epoch = check_point["epoch"]
        train_times = check_point["train_times"]
    else:
        logger.info(f"第一次训练Generator模型")

    # 这里需要做一个判断，判断是否有加载的模型
    if os.path.exists(D_LATEST_MODEL_PATH):
        # 获取上一次训练的检查点
        check_point = torch.load(D_LATEST_MODEL_PATH, map_location=device)
        logger.info(f"已存在训练的模型, epoch={check_point['epoch']}")
        # 加载上一次训练的模型的权重
        D.load_state_dict(check_point["model_state_dict"])
        # 加载上一次训练模型的优化器
        d_optimizer.load_state_dict(check_point["optimizer_state_dict"])
        d_loss_list = check_point["train_losses"]
        start_epoch = check_point["epoch"]
        train_times = check_point["train_times"]
    else:
        logger.info(f"第一次训练Discriminator模型")

    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"Epoch {epoch + 1}/{start_epoch + epochs}")
        g_avg_loss = 0
        d_avg_loss = 0
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            curr_batch_size = images.size(0)

            # 建立标签：1 代表真，0 代表假
            real_labels = torch.ones(curr_batch_size, 1).to(device)
            fake_labels = torch.zeros(curr_batch_size, 1).to(device)

            # ============================================
            # 训练判别器 D
            # ============================================
            # 1. 识别真图
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)

            # 2. 识别假图
            z = torch.randn(curr_batch_size, latent_size).to(device)  # 随机噪声
            fake_images = G(z)
            outputs = D(fake_images.detach())  # 注意用 detach，训练D时不更新G
            d_loss_fake = criterion(outputs, fake_labels)

            # 反向传播更新 D
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ============================================
            # 训练生成器 G
            # ============================================
            # 目标：让 D 认为假图是真图 (标签设为 1)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)

            # 反向传播更新 G
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_avg_loss = (g_avg_loss * i + g_loss.item()) / (i + 1)
            d_avg_loss = (d_avg_loss * i + d_loss.item()) / (i + 1)

        logger.info(f"Epoch [{epoch + 1}/{start_epoch + epochs}], d_loss: {d_avg_loss:.4f}, g_loss: {g_avg_loss:.4f}")
        g_loss_list.append(g_avg_loss)
        d_loss_list.append(d_avg_loss)
        # 每个 epoch 保存一次生成的图片看效果
        save_image(fake_images, f'samples/fake_images-{epoch + 1}.png')
    # torch 在保存模型的时候，应当将模型的参数，运行的次数，以及损失函数都保存进去，而不是仅仅保存参数
    torch.save(G.state_dict(), f=G_LATEST_MODEL_PATH)
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': G.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
        'train_losses': g_loss_list,
        'train_times': train_times + 1
    }, f=G_LATEST_MODEL_PATH)
    logger.info(f"模型保存成功，位置：{G_LATEST_MODEL_PATH}")
    # 绘制损失函数的图片
    plt.plot(g_loss_list)
    plt.savefig(MODEL_DIR / "g_loss.png")
    plt.close()

    torch.save(D.state_dict(), f=D_LATEST_MODEL_PATH)
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': D.state_dict(),
        'optimizer_state_dict': d_optimizer.state_dict(),
        'train_losses': d_loss_list,
        'train_times': train_times + 1
    }, f=D_LATEST_MODEL_PATH)
    logger.info(f"模型保存成功，位置：{D_LATEST_MODEL_PATH}")
    # 绘制损失函数的图片
    plt.plot(d_loss_list)
    plt.savefig(MODEL_DIR / "d_loss.png")


def test_model(num_samples=16):
    """
    g_path: 预训练生成器模型的路径 (如 'generator.pth')
    d_path: 预训练判别器模型的路径 (如 'discriminator.pth')
    num_samples: 想要生成的图片数量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 重新实例化网络结构 (确保这里的结构和训练时一模一样)
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 2. 加载保存好的权重参数
    g_check_point = torch.load(G_LATEST_MODEL_PATH, map_location=device)
    g_current_epoch = g_check_point["epoch"]
    logger.info(f"已存在训练的模型, epoch={g_check_point['epoch']}")
    # 加载上一次训练的模型的权重
    G.load_state_dict(g_check_point["model_state_dict"])

    d_check_point = torch.load(D_LATEST_MODEL_PATH, map_location=device)
    g_current_epoch = d_check_point["epoch"]
    logger.info(f"已存在训练的模型, epoch={d_check_point['epoch']}")
    # 加载上一次训练的模型的权重
    D.load_state_dict(d_check_point["model_state_dict"])

    # 3. 设置为评估模式 (Evaluation Mode)
    # 虽然现在的简单网络里没有 Dropout 或 BatchNorm，但这是必须养成的习惯
    G.eval()
    D.eval()

    # 4. 生成随机噪声
    # 维度是 [图片数量, 噪声长度]
    z = torch.randn(num_samples, latent_size).to(device)

    # 5. 推理过程 (不需要计算梯度)
    with torch.no_grad():
        # 让 G 生成图片
        fake_images = G(z)

        # 让 D 给这些生成的图片打个分 (可选)
        scores = D(fake_images)

    # 6. 保存或展示结果
    # normalize=True 会自动把 [-1, 1] 的像素值转回 [0, 1] 以便正确保存成图片
    save_image(fake_images, IMAGE_DIR / f'test_output_{g_current_epoch}.png', nrow=4, normalize=True)

    print(f"成功生成并保存了 {num_samples} 张图片到 f'test_output_{g_current_epoch}.png'")
    print(f"判别器对这批图片的平均评分是: {scores.mean().item():.4f} (越接近 0.5 说明生成越成功)")


if __name__ == "__main__":
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()
    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    else:
        logger.error("Please enter mode ('train' or 'test')")