# 导入库
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.logger import create_logger

logger = create_logger(__name__)
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# 1. 自定义数据集（假设你有 3000 个 .npy 文件）
# -----------------------------------
class PermeabilityDataset(Dataset):
    def __init__(self, data_dir, file_list, transform=None):
        self.data_dir = data_dir
        self.file_list = file_list  # ['model_0001.npy', ..., 'model_3000.npy']
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = f"{self.data_dir}/{self.file_list[idx]}"
        data = np.load(path)  # 假设 shape: (H, W, D) = (100, 100, 50)

        # 转为 tensor，增加 channel 维度: (1, H, W, D)
        data = torch.from_numpy(data).float().unsqueeze(0)

        if self.transform:
            data = self.transform(data)

        return data


# 示例：假设你有文件名列表
npy_file_dir: str = r"../../data/npy_files"
file_list = os.listdir(npy_file_dir)
dataset = PermeabilityDataset(data_dir=npy_file_dir, file_list=file_list)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # 小 batch，3D 数据大


# -----------------------------------
# 2. 3D VAE 模型
# -----------------------------------
class VAE3D(nn.Module):
    def __init__(self, input_shape=(16, 64, 64), latent_dim=64):
        super(VAE3D, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # 编码器：3D CNN 逐步降维
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),  # (1,100,100,50) → (32,50,50,25)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # → (64,25,25,12)
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # → (128,12,12,6)
            nn.ReLU(),
        )

        # 计算编码器输出的特征图大小
        with torch.no_grad():
            x = torch.randn(1, 1, *input_shape)
            h = self.encoder(x)
            self.h_shape = h.shape  # 用于 flatten 和 re-shape
            h_dim = h.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)

        # 解码器：3D 反卷积逐步升维
        self.fc_decode = nn.Linear(latent_dim, h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # (128,12,12,6) → (64,25,25,12)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # → (32,50,50,25)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),  # → (1,100,100,50)
            nn.Sigmoid()  # 输出在 [0,1]，后续可逆归一化
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, *self.h_shape[1:])  # 恢复 3D 形状
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


# -----------------------------------
# 3. 损失函数
# -----------------------------------
def vae_loss_3d(recon_x, x, mu, log_var, beta=1.0):
    # 重建误差：MSE（渗透率数据适合用 MSE）
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL 散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + beta * KLD  # beta 可调 KL 权重


# -----------------------------------
# 4. 训练代码
# -----------------------------------
model = VAE3D(latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)  # shape: [8, 1, 100, 100, 50]
        logger.info(f"data shape: {data.shape}:")
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = vae_loss_3d(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item() / len(data):.4f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')

# -----------------------------------
# 5. 生成新河道模型
# -----------------------------------
def generate_new_model(model, latent_dim=64, device='cuda'):
    """
    随机生成一个全新的 3D 渗透率模型
    """
    model.eval()
    with torch.no_grad():  # 不需要梯度
        # Step 1: 从标准正态分布随机采样 z
        z = torch.randn(1, latent_dim).to(device)  # shape: [1, 64]
        logger.info(f"生成随机数：{z}")
        # Step 2: 用解码器还原
        generated = model.decode(z)  # 输出 shape: [1, 1, H, W, D]

        # Step 3: 转为 numpy，去掉 batch 和 channel 维度
        generated = generated.cpu().squeeze().numpy()  # shape: [H, W, D]

        return generated

def test():
    # 现在加载训练好的模型
    model1 = VAE3D(latent_dim=64).to(device)
    model1.load_state_dict(torch.load('vae3d.pth'))
    logger.info(f"模型加载成功： {model1}")
    logger.info(f"开始生成模型")
    model1.eval()  # 切换到评估模式（关闭 dropout 等）
    new_permeability_model = generate_new_model(model, latent_dim=64)
    return new_permeability_model


# -----------------------------------
# 6. 开始训练
# -----------------------------------
if __name__ == "__main__":
    # for epoch in range(1, 5):
    #     train(epoch)
    # torch.save(model.state_dict(), "vae3d.pth")
    # logger.info(f"模型保存成功")
        # if epoch % 10 == 0:
        #     print("Generating a new permeability model...")
        #     new_model = generate_model()
        #     np.save(f"generated_model_epoch_{epoch}.npy", new_model)
        #     # 可视化某个切片
        #     plt.imshow(new_model[:, :, 25], cmap='viridis')
        #     plt.colorbar()
        #     plt.title(f"Generated Model - Slice at Z=25 (Epoch {epoch})")
        #     plt.show()

    a = test()
    print(a)
