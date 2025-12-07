import os
from pathlib import Path
import torch
from torch import Tensor
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import torch.nn.functional as F

from utils.logger import create_logger

logger = create_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARENT_DIR = Path(__file__).parent
MNIST_PARENT_DIR = PROJECT_ROOT / "data"
model_dir = PARENT_DIR / "models"
image_dir = PARENT_DIR / "images"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
model_path = model_dir / "diffusion_model.pth"
loss_image_path = image_dir / "loss_image.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 因为这里的 train=True , 所以读取的是训练集数据
# 如果想读取测试集数据，可以将 train=False
train_dataset = torchvision.datasets.MNIST(root=MNIST_PARENT_DIR, train=True, download=False,
                                           transform=torchvision.transforms.ToTensor())
# batch_size表示每个批次的样本数量， shuffle=True表示每个epoch打乱数据顺序
# 这里的 batch_size 可以根据显存大小进行调整
# 如果显存不够，可以将 batch_size 调小
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)



# ======================
# 轻量级 VAE for MNIST (潜空间 4x7x7)
# ======================
class VAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 4x4 → 我们不要这么小
        )
        # 调整：只下采样两次 → 7x7
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 14x14
            nn.ReLU(),
            nn.Conv2d(32, latent_dim * 2, 3, stride=2, padding=1),  # 7x7, 输出均值+方差
        )
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim * 7 * 7, 32 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # 28x28
            nn.Sigmoid()  # 输出 [0,1]
        )

    def encode(self, x):
        h = self.enc_conv(x)  # [B, 2*latent_dim, 7, 7]
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z.view(z.size(0), -1))
        h = h.view(-1, 32, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# ======================
# DDPM 超参数（用于潜空间）
# ======================
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 移到 device（稍后在 main 中赋值）
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None

# ======================
# 潜空间 UNet（带时间嵌入）
# ======================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



# ======================
# 简化版潜空间 UNet（无下采样，避免尺寸 mismatch）
# ======================
class SimpleLatentUNet(nn.Module):
    def __init__(self, latent_channels=4, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels + time_emb_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]
        t_emb = t_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_emb], dim=1)
        return self.net(x)

def q_sample(z_0: Tensor, t: Tensor, noise: Tensor = None):
    """在潜空间加噪"""
    if noise is None:
        noise = torch.randn_like(z_0)
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_prod * z_0 + sqrt_one_minus_alpha_prod * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    """从噪声生成潜变量，再解码"""
    model.eval()
    b = shape[0]
    z = torch.randn(shape, device=device)
    
    betas_t = betas.to(device)
    alphas_t = alphas.to(device)
    alphas_cumprod_t = alphas_cumprod.to(device)
    
    for i in reversed(range(T)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        noise_pred = model(z, t)
        
        alpha_t = alphas_t[t][:, None, None, None]
        alpha_cumprod_t_val = alphas_cumprod_t[t][:, None, None, None]
        beta_t = betas_t[t][:, None, None, None]
        
        if i == 0:
            noise = 0
        else:
            noise = torch.randn_like(z)
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            z - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t_val)) * noise_pred
        )
        variance = beta_t * ((1 - alpha_cumprod_t_val / alpha_t) / (1 - alpha_cumprod_t_val))
        z = mean + torch.sqrt(variance) * noise
        
    return z

def train_vae(vae, dataloader, epochs=10):
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in tqdm(dataloader, desc=f"VAE Epoch {epoch}"):
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = vae.loss_function(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"VAE Epoch {epoch}, Avg Loss: {total_loss/len(dataloader):.4f}")
    torch.save(vae.state_dict(), model_dir / "vae.pth")
    logger.info("VAE saved.")

def test_vae():
    """使用训练好的 VAE 进行重建和随机生成"""
    vae = VAE(latent_dim=4).to(device)
    vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
    vae.eval()

    mode = input("请选择：(1) 随机生成 (2) 重建对比: ")
    
    if mode == "1":
        # 随机生成：从 N(0,1) 采样潜变量
        with torch.no_grad():
            z = torch.randn(8, 4 * 7 * 7, device=device)  # 注意：VAE decoder 输入是展平的 latent
            # 或者更规范地：z = torch.randn(8, 4, 7, 7, device=device)
            # 但我们看你的 VAE.decode 接收的是 flatten 后的
            x_gen = vae.decode(z)
        x_gen = torch.clamp(x_gen, 0, 1).cpu()
        grid = torchvision.utils.make_grid(x_gen, nrow=4)
        save_path = image_dir / "vae_generated.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0), cmap="gray")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        logger.info(f"VAE 随机生成图像已保存至: {save_path}")

    elif mode == "2":
        # 重建对比
        x, _ = next(iter(train_dataloader))
        x = x[:8].to(device)
        with torch.no_grad():
            recon, _, _ = vae(x)
        x = torch.clamp(x.cpu(), 0, 1)
        recon = torch.clamp(recon.cpu(), 0, 1)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(torchvision.utils.make_grid(x, nrow=8)[0], cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(torchvision.utils.make_grid(recon, nrow=8)[0], cmap="gray")
        axes[1].set_title("VAE Reconstruction")
        for ax in axes: 
            ax.axis('off')
        save_path = image_dir / "vae_reconstruction.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"VAE 重建对比图已保存至: {save_path}")


def train_ldm(vae, unet, dataloader, epochs=20):
    vae.eval()  # 冻结 VAE
    unet.train()
    optimizer = torch.optim.Adam(unet.parameters(), lr=2e-4)
    
    start_epoch = 0
    ldm_path = model_dir / "ldm_unet.pth"
    if ldm_path.exists():
        ckpt = torch.load(ldm_path, map_location=device)
        unet.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        logger.info(f"Resuming LDM from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        for x, _ in tqdm(dataloader, desc=f"LDM Epoch {epoch}"):
            x = x.to(device)
            with torch.no_grad():
                mu, _ = vae.encode(x)
                z_0 = mu  # 使用确定性编码（不加随机性）
            batch_size = z_0.shape[0]
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(z_0)
            z_noisy = q_sample(z_0, t, noise)
            noise_pred = unet(z_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"LDM Epoch {epoch}, Avg Loss: {total_loss/len(dataloader):.6f}")
    
    torch.save({
        'epoch': start_epoch + epochs,
        'model': unet.state_dict(),
        'optimizer': optimizer.state_dict()
    }, ldm_path)
    logger.info("LDM saved.")

def test_ldm():
    # 加载模型
    vae = VAE(latent_dim=4).to(device)
    vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
    vae.eval()

    unet = SimpleLatentUNet(latent_channels=4).to(device)  # ← 改这里
    ckpt = torch.load(model_dir / "ldm_unet.pth", map_location=device)
    unet.load_state_dict(ckpt['model'])
    unet.eval()

    mode = input("请选择：(1) 随机生成 (2) 重建对比: ")
    
    if mode == "1":
        # 随机生成
        z_T = torch.randn(8, 4, 7, 7).to(device)
        z_0 = p_sample_loop(unet, z_T.shape)
        with torch.no_grad():
            x_gen = vae.decode(z_0)
        x_gen = torch.clamp(x_gen, 0, 1).cpu()
        grid = torchvision.utils.make_grid(x_gen, nrow=4)
        save_path = image_dir / "ldm_generated.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0), cmap="gray")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        logger.info(f"随机生成图像已保存至: {save_path}")

    elif mode == "2":
        # 重建对比
        x, _ = next(iter(train_dataloader))
        x = x[:8].to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            z_0 = mu
            t = torch.full((8,), 500, device=device, dtype=torch.long)  # 中等噪声
            noise = torch.randn_like(z_0)
            z_noisy = q_sample(z_0, t, noise)
            z_recon = p_sample_loop_from_t(unet, z_noisy, t[0].item())  # 从 t=500 开始去噪
            x_recon = vae.decode(z_recon)
        
        x = torch.clamp(x.cpu(), 0, 1)
        x_recon = torch.clamp(x_recon.cpu(), 0, 1)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(torchvision.utils.make_grid(x, nrow=8)[0], cmap="gray")
        axes[1].imshow(torchvision.utils.make_grid(x_recon, nrow=8)[0], cmap="gray")
        for ax in axes: ax.axis('off')
        save_path = image_dir / "ldm_reconstruction.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"重建对比图已保存至: {save_path}")


# 辅助函数：从任意 t 开始去噪
@torch.no_grad()
def p_sample_loop_from_t(model, z_t, start_t):
    model.eval()
    z = z_t
    for i in reversed(range(start_t)):
        t = torch.full((z.shape[0],), i, device=device, dtype=torch.long)
        noise_pred = model(z, t)
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        if i == 0:
            noise = 0
        else:
            noise = torch.randn_like(z)
        mean = (1 / torch.sqrt(alpha_t)) * (
            z - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
        )
        variance = beta_t * ((1 - alpha_cumprod_t / alpha_t) / (1 - alpha_cumprod_t))
        z = mean + torch.sqrt(variance) * noise
    return z

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["vae", "vae_test","ldm", "test"], required=True)
    args = parser.parse_args()

    # 初始化调度器（移到 device）
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    if args.stage == "vae":
        vae = VAE(latent_dim=4).to(device)
        train_vae(vae, train_dataloader, epochs=10)
    elif args.stage == "ldm":
        vae = VAE(latent_dim=4).to(device)
        vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
        unet = SimpleLatentUNet(latent_channels=4).to(device)  # ← 改这里
        train_ldm(vae, unet, train_dataloader, epochs=20)
    elif args.stage == "test":
        test_ldm()
    elif args.stage == "vae_test":      # ← 新增
        test_vae()