import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


# 数据集类
class CrackDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                          f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image


# VAE模型
class VAE(nn.Module):
    def __init__(self, img_size=256, z_dim=128):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算编码器输出大小
        self.encoder_output_size = 512 * 8 * 8

        self.fc_mu = nn.Linear(self.encoder_output_size, z_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, z_dim)

        # 解码器
        self.decoder_input = nn.Linear(z_dim, self.encoder_output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim).to(device)
        samples = self.decode(z)
        return samples


# 扩散模型（在潜在空间）
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 噪声预测网络
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 定义噪声调度
        self.betas = self._linear_beta_schedule(num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 注册为缓冲区，这样它们会随着模型移动到设备上
        self.register_buffer('betas_buffer', self.betas)
        self.register_buffer('alphas_buffer', self.alphas)
        self.register_buffer('alphas_cumprod_buffer', self.alphas_cumprod)

    def _linear_beta_schedule(self, num_timesteps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, num_timesteps)

    def forward(self, z, t):
        # 将时间步转换为浮点类型
        t = t.float()

        # 时间步嵌入
        t_embed = self.time_embed(t.view(-1, 1))

        # 预测噪声
        noise_pred = self.noise_predictor(torch.cat([z, t_embed], dim=1))
        return noise_pred

    def q_sample(self, z_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z_start)

        # 使用注册的缓冲区，确保在同一设备上
        sqrt_alphas_cumprod_t = self.alphas_cumprod_buffer[t] ** 0.5
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod_buffer[t]) ** 0.5

        # 扩展维度以匹配 z_start 的形状
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * z_start + sqrt_one_minus_alphas_cumprod_t * noise


# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# 扩散模型损失函数
def diffusion_loss(model, z_0, t):
    noise = torch.randn_like(z_0)
    z_t = model.q_sample(z_0, t, noise)
    noise_pred = model(z_t, t)
    return F.mse_loss(noise_pred, noise)


# 加载已保存的VAE模型
def load_vae_model(model_path, device, z_dim=128):
    vae_model = VAE(z_dim=z_dim).to(device)
    vae_model.load_state_dict(torch.load(model_path, map_location=device))
    vae_model.eval()
    print(f"已加载VAE模型: {model_path}")
    return vae_model


# 验证VAE
def validate_vae(vae_model, dataloader, device, num_samples=5):
    vae_model.eval()

    # 从数据集中获取一些样本
    data_iter = iter(dataloader)
    original_imgs = next(data_iter)[:num_samples].to(device)

    with torch.no_grad():
        # 重建图像
        reconstructed, _, _ = vae_model(original_imgs)

        # 从潜在空间采样生成新图像
        generated_imgs = vae_model.sample(num_samples, device)

    # 显示结果
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))

    for i in range(num_samples):
        # 原始图像
        orig_img = original_imgs[i].cpu().squeeze()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'Original {i + 1}')
        axes[0, i].axis('off')

        # 重建图像
        recon_img = reconstructed[i].cpu().squeeze()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i + 1}')
        axes[1, i].axis('off')

        # 生成图像
        gen_img = generated_imgs[i].cpu().squeeze()
        axes[2, i].imshow(gen_img, cmap='gray')
        axes[2, i].set_title(f'Generated {i + 1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('vae_validation.png')
    plt.show()

    vae_model.train()


# 训练扩散模型
def train_diffusion(vae_model, dataloader, device, num_epochs=500, lr=1e-4):
    # 冻结VAE的权重
    for param in vae_model.parameters():
        param.requires_grad = False

    # 创建扩散模型并移动到设备
    diffusion_model = DiffusionModel(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    print("训练扩散模型...")
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)

            # 通过VAE编码器获取潜在表示
            with torch.no_grad():
                mu, logvar = vae_model.encode(data)
                z_0 = vae_model.reparameterize(mu, logvar)

            # 随机时间步
            t = torch.randint(0, diffusion_model.num_timesteps, (z_0.size(0),), device=device)

            # 扩散模型训练
            loss = diffusion_loss(diffusion_model, z_0, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if epoch % 50 == 0:
            avg_loss = total_loss / batch_count
            print(f"Diffusion Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # 保存扩散模型
    torch.save(diffusion_model.state_dict(), "crack_diffusion.pth")
    print("扩散模型已保存: crack_diffusion.pth")

    return diffusion_model


# 从扩散模型生成样本
def generate_from_diffusion(vae_model, diffusion_model, device, num_samples=5):
    vae_model.eval()
    diffusion_model.eval()

    # 扩散模型采样过程
    with torch.no_grad():
        # 从纯噪声开始
        z_t = torch.randn(num_samples, 128, device=device)

        # 反向扩散过程
        for t in reversed(range(diffusion_model.num_timesteps)):
            # 创建时间步张量
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = diffusion_model(z_t, t_tensor)

            # 计算系数 - 使用注册的缓冲区
            alpha_t = diffusion_model.alphas_buffer[t]
            alpha_cumprod_t = diffusion_model.alphas_cumprod_buffer[t]
            beta_t = diffusion_model.betas_buffer[t]

            # 扩展维度以匹配 z_t 的形状
            alpha_t = alpha_t.view(1, 1)
            alpha_cumprod_t = alpha_cumprod_t.view(1, 1)
            beta_t = beta_t.view(1, 1)

            if t > 0:
                noise = torch.randn_like(z_t)
            else:
                noise = 0

            # 更新z_t
            z_t = (1 / torch.sqrt(alpha_t)) * (
                    z_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise

        # 通过VAE解码器生成图像
        generated_imgs = vae_model.decode(z_t)

    # 显示生成的图像
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        img = generated_imgs[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'LDM Generated {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('ldm_generated.png')
    plt.show()

    vae_model.train()
    diffusion_model.train()


# 加载扩散模型
def load_diffusion_model(model_path, device, latent_dim=128):
    diffusion_model = DiffusionModel(latent_dim=latent_dim).to(device)
    diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion_model.eval()
    print(f"已加载扩散模型: {model_path}")
    return diffusion_model


# 主函数 - 从保存的模型继续
def main_from_checkpoint():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_dir = 'images'
    dataset = CrackDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 1. 加载已保存的VAE模型
    vae_model = load_vae_model("crack_vae.pth", device)

    # 2. 验证VAE
    print("验证VAE模型...")
    validate_vae(vae_model, dataloader, device)

    # 3. 检查是否有已保存的扩散模型
    diffusion_model_path = "crack_diffusion.pth"
    if os.path.exists(diffusion_model_path):
        # 加载已保存的扩散模型
        diffusion_model = load_diffusion_model(diffusion_model_path, device)
        print("加载已保存的扩散模型")
    else:
        # 训练新的扩散模型
        print("训练新的扩散模型...")
        diffusion_model = train_diffusion(vae_model, dataloader, device, num_epochs=500)

    # 4. 从扩散模型生成样本
    print("从扩散模型生成样本...")
    generate_from_diffusion(vae_model, diffusion_model, device, num_samples=5)


# 完整训练流程（从头开始）
def main_full_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_dir = 'images'
    dataset = CrackDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 1. 训练VAE
    print("训练VAE...")
    vae_model = VAE(z_dim=128).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

    for epoch in range(100):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            recon, mu, logvar = vae_model(data)
            loss = vae_loss(recon, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # 保存VAE模型
    torch.save(vae_model.state_dict(), "crack_vae.pth")
    print("VAE模型已保存: crack_vae.pth")

    # 后续步骤
    main_from_checkpoint()


# 选择运行模式
if __name__ == "__main__":
    # 如果已经有VAE模型，直接使用检查点继续
    if os.path.exists("crack_vae.pth"):
        print("检测到已保存的VAE模型，从检查点继续...")
        main_from_checkpoint()
    else:
        print("未找到已保存的模型，开始完整训练...")
        main_full_training()