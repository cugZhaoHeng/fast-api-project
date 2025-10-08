import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 自定义数据集类 - 从新的npy文件加载（白色背景）
class DFNDatasetWhiteBg(Dataset):
    def __init__(self, npy_file_path):
        """
        从npy文件加载DFN图像数据（白色背景，黑色裂缝）
        """
        self.images = np.load(npy_file_path)
        print(f"从 {npy_file_path} 加载数据，形状: {self.images.shape}")

        # 转换为tensor并添加通道维度
        self.images = torch.tensor(self.images, dtype=torch.float32).unsqueeze(1)
        print(f"处理后形状: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# 改进的VAE模型 - 使用更深的网络和残差连接
class ImprovedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ImprovedResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果需要使用注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 应用注意力机制
        if self.use_attention:
            attention_weights = self.attention(out)
            out = out * attention_weights

        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 改进的编码器
class ImprovedVAE_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE_Encoder, self).__init__()
        self.latent_dim = latent_dim

        # 更深的编码器网络
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        )

        self.encoder_blocks = nn.Sequential(
            ImprovedResNetBlock(64, 128),  # 16x16
            ImprovedResNetBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            ImprovedResNetBlock(256, 256, use_attention=True),  # 8x8
            ImprovedResNetBlock(256, 512),
            nn.AdaptiveAvgPool2d((4, 4))  # 8x8 -> 4x4
        )

        # 潜在空间的均值和方差
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder_blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# 改进的解码器
class ImprovedVAE_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE_Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder_blocks = nn.Sequential(
            ImprovedResNetBlock(512, 512, use_attention=True),  # 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            ImprovedResNetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            ImprovedResNetBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            ImprovedResNetBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)
        return self.decoder_blocks(h)


# 改进的完整VAE模型
class ImprovedVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ImprovedVAE_Encoder(latent_dim)
        self.decoder = ImprovedVAE_Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# 改进的损失函数 - 结合多种损失
class ImprovedVAELoss(nn.Module):
    def __init__(self, lambda_kl=1e-6, lambda_mse=1.0, lambda_ssim=0.1):
        super(ImprovedVAELoss, self).__init__()
        self.lambda_kl = lambda_kl
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim

    def ssim_loss(self, x, y, window_size=11, size_average=True):
        """计算SSIM损失"""
        # 简化版SSIM计算
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2, count_include_pad=False)
        mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2, count_include_pad=False)

        sigma_x = F.avg_pool2d(x ** 2, window_size, 1, window_size // 2, count_include_pad=False) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, window_size, 1, window_size // 2, count_include_pad=False) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2, count_include_pad=False) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim = ssim_n / ssim_d

        if size_average:
            return 1 - ssim.mean()
        else:
            return 1 - ssim.view(ssim.size(0), -1).mean(1)

    def forward(self, recon_x, x, mu, logvar):
        # 重建损失 - 多种损失组合
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        mse_loss = F.mse_loss(recon_x, x, reduction='sum')
        ssim_loss = self.ssim_loss(recon_x, x) * x.numel()

        # 组合重建损失
        recon_loss = bce_loss + self.lambda_mse * mse_loss + self.lambda_ssim * ssim_loss

        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失
        total_loss = recon_loss + self.lambda_kl * kl_loss

        return total_loss, recon_loss, kl_loss


# 改进的训练函数
def train_improved_vae(model, dataloader, optimizer, scheduler, epochs=100):
    model.train()
    train_losses = []
    recon_losses = []
    kl_losses = []

    # 使用改进的损失函数
    criterion = ImprovedVAELoss(lambda_kl=1e-6, lambda_mse=1.0, lambda_ssim=0.1)

    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = model(batch)

            # 计算损失
            loss, recon_loss, kl_loss = criterion(recon_batch, batch, mu, logvar)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = total_recon / len(dataloader.dataset)
        avg_kl = total_kl / len(dataloader.dataset)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        # 更新学习率
        scheduler.step(avg_loss)

        # 手动打印学习率变化信息
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, LR: {current_lr:.2e}")

        # 每20个epoch保存一次模型和生成示例
        if (epoch + 1) % 20 == 0:
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f'improved_vae_white_epoch_{epoch + 1}.pth')

            # 生成示例
            generate_and_show_samples(model, num_samples=10, epoch=epoch + 1)

    return train_losses, recon_losses, kl_losses


# 生成并显示样本的函数
def generate_and_show_samples(model, num_samples=10, epoch=None):
    model.eval()

    with torch.no_grad():
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, model.latent_dim).to(device)

        # 通过解码器生成新图像
        generated_imgs = model.decoder(z)

        # 转换为numpy数组
        generated_imgs = generated_imgs.cpu().numpy()

        # 显示生成的图像
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(generated_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
            title = f'Generated {i + 1}'
            if epoch:
                title += f' (Epoch {epoch})'
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    model.train()


# 验证改进的VAE重建效果
def validate_improved_reconstruction(model, dataset, num_samples=10):
    model.eval()

    with torch.no_grad():
        # 随机选择样本
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        original_imgs = torch.stack([dataset[i] for i in indices]).to(device)

        # 重建图像
        recon_batch, _, _ = model(original_imgs)

        # 转换为numpy数组用于显示
        original_imgs_np = original_imgs.cpu().numpy()
        recon_imgs_np = recon_batch.cpu().numpy()

        # 显示原始和重建图像
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))

        for i in range(num_samples):
            # 原始图像
            axes[0, i].imshow(original_imgs_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # 重建图像
            axes[1, i].imshow(recon_imgs_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Reconstructed {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

        # 计算重建误差
        mse = np.mean((original_imgs_np - recon_imgs_np) ** 2)
        print(f"平均重建MSE: {mse:.6f}")


# 从潜在空间生成高质量DFN图像
def generate_high_quality_dfns(model, num_samples=10, temperature=1.0):
    """
    生成高质量DFN图像

    Args:
        model: 训练好的VAE模型
        num_samples: 生成样本数量
        temperature: 采样温度，控制多样性
    """
    model.eval()

    with torch.no_grad():
        # 从标准正态分布采样潜在向量
        z = torch.randn(num_samples, model.latent_dim).to(device) * temperature

        # 通过解码器生成新图像
        generated_imgs = model.decoder(z)

        # 转换为numpy数组
        generated_imgs = generated_imgs.cpu().numpy()

        # 显示生成的图像
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(generated_imgs[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'High Quality DFN {i + 1}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        return generated_imgs


# 主训练流程
# 检查新的npy文件是否存在
npy_file_path_white = 'dfn_images_white_bg.npy'
if not os.path.exists(npy_file_path_white):
    print(f"未找到 {npy_file_path_white}，请先运行第一部分代码生成数据")
    exit()

# 创建数据集和数据加载器
print("创建白色背景数据集...")
dataset_white = DFNDatasetWhiteBg(npy_file_path_white)
dataloader_white = DataLoader(dataset_white, batch_size=32, shuffle=True)

# 可视化一些样本
print("可视化白色背景样本...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    sample = dataset_white[i].squeeze().numpy()
    ax.imshow(sample, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'白底样本 {i + 1}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 初始化改进的模型和优化器
latent_dim = 128
improved_vae_white = ImprovedVAE(latent_dim=latent_dim).to(device)
optimizer_white = optim.Adam(improved_vae_white.parameters(), lr=1e-4, weight_decay=1e-5)

# 兼容不同PyTorch版本的调度器
try:
    scheduler_white = optim.lr_scheduler.ReduceLROnPlateau(optimizer_white, mode='min', factor=0.5, patience=10,
                                                           verbose=True)
except TypeError:
    print("检测到旧版PyTorch，使用兼容模式")
    scheduler_white = optim.lr_scheduler.ReduceLROnPlateau(optimizer_white, mode='min', factor=0.5, patience=10)

print(f"改进VAE模型参数量: {sum(p.numel() for p in improved_vae_white.parameters())}")

# 训练改进的VAE（使用白色背景数据）
print("开始训练改进的VAE（白色背景）...")
train_losses_white, recon_losses_white, kl_losses_white = train_improved_vae(
    improved_vae_white, dataloader_white, optimizer_white, scheduler_white, epochs=100
)

# 绘制训练损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses_white)
plt.title('Total Loss (White BG)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(recon_losses_white)
plt.title('Reconstruction Loss (White BG)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(kl_losses_white)
plt.title('KL Loss (White BG)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# 验证改进的VAE重建效果（白色背景）
print("验证改进的VAE重建效果（白色背景）...")
validate_improved_reconstruction(improved_vae_white, dataset_white)

# 从潜在空间生成新的DFN图像（白色背景）
print("从潜在空间生成新的DFN图像（白色背景）...")
generated_dfns_white = generate_high_quality_dfns(improved_vae_white, temperature=0.7)

# 保存训练好的VAE模型（白色背景）
torch.save({
    'model_state_dict': improved_vae_white.state_dict(),
    'optimizer_state_dict': optimizer_white.state_dict(),
    'latent_dim': latent_dim,
    'train_losses': train_losses_white
}, 'improved_vae_white_bg_final.pth')

print("白色背景VAE模型已保存为 'improved_vae_white_bg_final.pth'")