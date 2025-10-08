import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 首先，我们需要重新定义VAE模型结构（与第二部分相同）
class ImprovedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ImprovedResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        if self.use_attention:
            attention_weights = self.attention(out)
            out = out * attention_weights

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImprovedVAE_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE_Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.encoder_blocks = nn.Sequential(
            ImprovedResNetBlock(64, 128),
            ImprovedResNetBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ImprovedResNetBlock(256, 256, use_attention=True),
            ImprovedResNetBlock(256, 512),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder_blocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ImprovedVAE_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedVAE_Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder_blocks = nn.Sequential(
            ImprovedResNetBlock(512, 512, use_attention=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            ImprovedResNetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            ImprovedResNetBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ImprovedResNetBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)
        return self.decoder_blocks(h)


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


# 自定义数据集类
class DFNDatasetWhiteBg(Dataset):
    def __init__(self, npy_file_path):
        self.images = np.load(npy_file_path)
        print(f"从 {npy_file_path} 加载数据，形状: {self.images.shape}")
        self.images = torch.tensor(self.images, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# 加载白色背景的VAE模型
def load_vae_model_white_bg(model_path, latent_dim=128):
    """加载训练好的白色背景VAE模型"""
    model = ImprovedVAE(latent_dim=latent_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# 定义扩散模型的时间步调度器
class BetaScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear', device='cpu'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.device = device

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x, t):
        """根据时间步t向数据添加噪声"""
        # 确保t在正确的设备上
        t = t.to(self.device)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        return noisy_x, noise


# 定义U-Net模型用于扩散过程
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )

        self.residual_connection = nn.Linear(in_channels,
                                             out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(1)
        h = self.block2(h)
        return h + self.residual_connection(x)


class UNet1D(nn.Module):
    def __init__(self, latent_dim=128, time_emb_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.down1 = ResidualBlock(latent_dim, 256, time_emb_dim)
        self.down2 = ResidualBlock(256, 512, time_emb_dim)
        self.down3 = ResidualBlock(512, 512, time_emb_dim)

        self.mid = ResidualBlock(512, 512, time_emb_dim)

        self.up1 = ResidualBlock(512 + 512, 512, time_emb_dim)
        self.up2 = ResidualBlock(512 + 512, 256, time_emb_dim)
        self.up3 = ResidualBlock(256 + 256, 128, time_emb_dim)

        self.out = nn.Sequential(
            nn.Linear(128, latent_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h1 = self.down1(x, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)

        h_mid = self.mid(h3, t_emb)

        h = self.up1(torch.cat([h_mid, h3], dim=-1), t_emb)
        h = self.up2(torch.cat([h, h2], dim=-1), t_emb)
        h = self.up3(torch.cat([h, h1], dim=-1), t_emb)

        return self.out(h)


# 定义LDM训练器
class LDMTrainer:
    def __init__(self, vae_model, latent_dim=128, timesteps=1000):
        self.vae_model = vae_model
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.unet = UNet1D(latent_dim=latent_dim).to(device)
        # 修复：将设备传递给BetaScheduler
        self.scheduler = BetaScheduler(timesteps=timesteps, schedule_type='cosine', device=device)
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=1e-4)

        print(f"LDM U-Net参数量: {sum(p.numel() for p in self.unet.parameters())}")

    def train_step(self, x):
        """训练单步"""
        self.unet.train()

        with torch.no_grad():
            mu, logvar = self.vae_model.encoder(x)
            z = self.vae_model.reparameterize(mu, logvar)

        # 确保时间步t在正确的设备上
        t = torch.randint(0, self.timesteps, (z.shape[0],), device=device).long()
        noisy_z, noise = self.scheduler.add_noise(z, t)
        predicted_noise = self.unet(noisy_z, t)

        loss = F.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples=1, steps=100):
        """从LDM生成样本"""
        self.unet.eval()

        z = torch.randn(num_samples, self.latent_dim, device=device)

        for i in tqdm(reversed(range(0, steps)), desc="采样"):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            predicted_noise = self.unet(z, t)

            alpha = self.scheduler.alphas_cumprod[i]
            alpha_prev = self.scheduler.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)

            z = (1 / torch.sqrt(alpha)) * (
                    z - ((1 - alpha) / torch.sqrt(1 - self.scheduler.alphas_cumprod[i])) * predicted_noise
            )

            if i > 0:
                sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                z = z + sigma * torch.randn_like(z)

        with torch.no_grad():
            generated_images = self.vae_model.decoder(z)

        return generated_images.cpu().numpy()


# 训练LDM模型
def train_ldm(trainer, dataloader, epochs=100):
    """训练LDM模型"""
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'unet_state_dict': trainer.unet.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': avg_loss
            }, f'ldm_model_white_epoch_{epoch + 1}.pth')

            generate_ldm_samples(trainer, num_samples=10, epoch=epoch + 1)

    return losses


# 生成LDM样本并显示
def generate_ldm_samples(trainer, num_samples=10, epoch=None):
    """生成并显示LDM样本"""
    generated_images = trainer.sample(num_samples=num_samples, steps=100)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(generated_images[i, 0], cmap='gray', vmin=0, vmax=1)
        title = f'LDM Generated {i + 1}'
        if epoch:
            title += f' (Epoch {epoch})'
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 批量生成DFN图像
def batch_generate_dfns(trainer, num_batches=10, batch_size=32):
    """批量生成DFN图像"""
    print(f"批量生成 {num_batches * batch_size} 个DFN图像...")

    all_generated_images = []

    for i in range(num_batches):
        print(f"生成批次 {i + 1}/{num_batches}")
        generated_images = trainer.sample(num_samples=batch_size, steps=100)
        all_generated_images.append(generated_images)

    all_generated_images = np.concatenate(all_generated_images, axis=0)

    print(f"生成的DFN图像形状: {all_generated_images.shape}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        axes[i].imshow(all_generated_images[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Batch Generated {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    np.save('ldm_generated_dfns_white.npy', all_generated_images)
    print(f"生成的DFN图像已保存为 'ldm_generated_dfns_white.npy'")

    return all_generated_images


# 验证生成的DFN统计特性
def validate_generated_dfns_statistics(generated_images, original_dataset, vae_model):
    """验证生成的DFN统计特性"""
    from sklearn.decomposition import PCA

    print("验证生成的DFN统计特性...")

    original_sample = original_dataset[:100].numpy()
    mse = np.mean((original_sample - generated_images[:100]) ** 2)
    print(f"与原始样本的平均MSE: {mse:.6f}")

    with torch.no_grad():
        original_batch = original_dataset[:100].to(device)
        mu_original, _ = vae_model.encoder(original_batch)
        z_original = vae_model.reparameterize(mu_original, torch.zeros_like(mu_original))

        generated_batch = torch.tensor(generated_images[:100], dtype=torch.float32).to(device)
        mu_generated, _ = vae_model.encoder(generated_batch)
        z_generated = vae_model.reparameterize(mu_generated, torch.zeros_like(mu_generated))

        z_original_np = z_original.cpu().numpy()
        z_generated_np = z_generated.cpu().numpy()

        pca = PCA(n_components=2)
        combined_z = np.concatenate([z_original_np, z_generated_np], axis=0)
        combined_2d = pca.fit_transform(combined_z)

        original_2d = combined_2d[:100]
        generated_2d = combined_2d[100:]

        plt.figure(figsize=(10, 8))
        plt.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.7, label='Original')
        plt.scatter(generated_2d[:, 0], generated_2d[:, 1], alpha=0.7, label='Generated')
        plt.title('潜在空间分布比较')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()

        print(f"潜在空间方差解释比例: {pca.explained_variance_ratio_.sum():.4f}")

    generated_flatten = generated_images.reshape(generated_images.shape[0], -1)
    distances = []
    for i in range(min(100, len(generated_flatten))):
        for j in range(i + 1, min(100, len(generated_flatten))):
            dist = np.linalg.norm(generated_flatten[i] - generated_flatten[j])
            distances.append(dist)

    avg_distance = np.mean(distances)
    print(f"生成图像之间的平均距离: {avg_distance:.4f}")

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(20):
        ax = axes[i // 5, i % 5]
        ax.imshow(generated_images[i, 0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Generated DFN {i + 1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# 主训练流程
# 检查数据文件是否存在
npy_file_path_white = 'dfn_images_white_bg.npy'
if not os.path.exists(npy_file_path_white):
    print(f"未找到 {npy_file_path_white}，请先运行第一部分代码生成数据")
    exit()

# 检查VAE模型文件是否存在
vae_model_path = 'improved_vae_white_bg_final.pth'
if not os.path.exists(vae_model_path):
    print(f"未找到 {vae_model_path}，请先运行第二部分代码训练VAE模型")
    exit()

# 创建数据集
print("创建白色背景数据集...")
dataset_white = DFNDatasetWhiteBg(npy_file_path_white)
dataloader_white = DataLoader(dataset_white, batch_size=32, shuffle=True)

# 加载白色背景VAE模型
print("加载白色背景VAE模型...")
vae_model_white = load_vae_model_white_bg(vae_model_path, latent_dim=128)

# 初始化LDM训练器
print("初始化LDM训练器（白色背景）...")
ldm_trainer_white = LDMTrainer(vae_model_white, latent_dim=128, timesteps=1000)

# 训练LDM模型
print("开始训练LDM（白色背景）...")
ldm_losses_white = train_ldm(ldm_trainer_white, dataloader_white, epochs=100)

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(ldm_losses_white)
plt.title('LDM Training Loss (White Background)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 批量生成白色背景DFN图像
print("批量生成白色背景DFN图像...")
batch_generated_dfns_white = batch_generate_dfns(ldm_trainer_white, num_batches=10, batch_size=32)

# 验证生成的DFN统计特性
validate_generated_dfns_statistics(batch_generated_dfns_white, dataset_white, vae_model_white)

# 保存最终LDM模型
torch.save({
    'unet_state_dict': ldm_trainer_white.unet.state_dict(),
    'optimizer_state_dict': ldm_trainer_white.optimizer.state_dict(),
    'latent_dim': ldm_trainer_white.latent_dim,
    'timesteps': ldm_trainer_white.timesteps,
    'train_losses': ldm_losses_white
}, 'ldm_model_white_bg_final.pth')

print("白色背景LDM模型已保存为 'ldm_model_white_bg_final.pth'")
print("白色背景LDM训练和生成完成！")