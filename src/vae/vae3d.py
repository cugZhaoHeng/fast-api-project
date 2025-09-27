# 导入库
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.logger import create_logger
import matplotlib

# 设置中文字体和负号显示（避免乱码）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = create_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# -----------------------------------
# 1. 自定义数据集（添加数据集划分）
# -----------------------------------
class PermeabilityDataset(Dataset):
    def __init__(self, data_dir, file_list, transform=None):
        self.data_dir = data_dir
        self.file_list = file_list  # ['model_0001.npy', ...]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(path)  # (H, W, D)
        
        # 转为 tensor，增加 channel 维度: (1, H, W, D)
        data = torch.from_numpy(data).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
            
        return data

# 数据集划分：2000训练 + 242测试
npy_file_dir: Path = PROJECT_ROOT / "data" / "npy_files"
all_files = os.listdir(npy_file_dir)
np.random.seed(42)  # 保证可复现
np.random.shuffle(all_files)  # 随机打乱

# 2000训练, 242测试 (2242 total)
train_files = all_files[:2000]
test_files = all_files[2000:]

# 创建数据集
train_dataset = PermeabilityDataset(data_dir=npy_file_dir, file_list=train_files)
test_dataset = PermeabilityDataset(data_dir=npy_file_dir, file_list=test_files)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -----------------------------------
# 2. 3D VAE 模型（保持不变）
# -----------------------------------
class VAE3D(nn.Module):
    def __init__(self, input_shape=(16, 64, 64), latent_dim=64):
        super(VAE3D, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.randn(1, 1, *input_shape)
            h = self.encoder(x)
            self.h_shape = h.shape
            h_dim = h.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
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
        h = h.view(-1, *self.h_shape[1:])
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# -----------------------------------
# 3. 修改后的损失函数（添加perceptual loss）
# -----------------------------------
def vae_loss_3d(recon_x, x, mu, log_var, beta=1.0, gamma=0.1):
    """添加perceptual loss（使用VAE编码器前两层作为感知网络）"""
    # 1. 重构损失 (MSE)
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 3. Perceptual Loss (使用VAE编码器前两层)
    # 注意：这里使用模型自身的编码器作为感知网络
    feat_real = model.encoder[:2](x)  # 提取真实数据的特征
    feat_recon = model.encoder[:2](recon_x)  # 提取重建数据的特征
    perceptual_loss = nn.functional.mse_loss(feat_real, feat_recon)
    
    # 4. 总损失
    total_loss = BCE + beta * KLD + gamma * perceptual_loss
    return total_loss, BCE, KLD, perceptual_loss

# -----------------------------------
# 4. 训练代码（添加损失记录和保存）
# -----------------------------------
model = VAE3D(latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 用于记录训练/测试损失
train_losses = []
test_losses = []
epoch_losses = {"train": [], "test": []}

def train(epoch):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0
    total_perceptual = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss, bce, kld, perceptual = vae_loss_3d(recon_batch, data, mu, log_var)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
        total_perceptual += perceptual.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader.dataset)
    avg_bce = total_bce / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)
    avg_perceptual = total_perceptual / len(train_loader.dataset)
    
    train_losses.append(avg_loss)
    epoch_losses["train"].append((avg_loss, avg_bce, avg_kld, avg_perceptual))
    
    logger.info(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f} '
                f'(BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}, Perceptual: {avg_perceptual:.4f})')

def test(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss, _, _, _ = vae_loss_3d(recon_batch, data, mu, log_var)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    test_losses.append(avg_loss)
    epoch_losses["test"].append(avg_loss)
    
    logger.info(f'====> Test Epoch: {epoch} Average test loss: {avg_loss:.4f}')

def save_loss_curve():
    """保存损失曲线图片"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(test_losses, 'r--', label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    logger.info("训练损失曲线已保存: loss_curve.png")

# -----------------------------------
# 5. 训练主循环
# -----------------------------------
if __name__ == "__main__":
    num_epochs = 50  # 增加训练轮数
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)
        
        # 每5轮保存一次模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"vae3d_epoch_{epoch}.pth")
            logger.info(f"模型已保存: vae3d_epoch_{epoch}.pth")
    
    # 保存最终模型和损失曲线
    torch.save(model.state_dict(), "vae3d_final.pth")
    save_loss_curve()
    logger.info("训练完成，模型和损失曲线已保存")