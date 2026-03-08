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
import json

from utils.logger import create_logger

logger = create_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARENT_DIR = Path(__file__).parent
MNIST_PARENT_DIR = PROJECT_ROOT / "data"
model_dir = PARENT_DIR / "models"
image_dir = PARENT_DIR / "images"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset
train_dataset = torchvision.datasets.MNIST(
    root=MNIST_PARENT_DIR, train=True, download=False,
    transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # 增大 batch_size

# ======================
# VAE (latent_dim=8 for better quality)
# ======================
class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 28 -> 14 -> 7
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 14x14
            nn.ReLU(),
            nn.Conv2d(32, latent_dim * 2, 3, stride=2, padding=1),  # 7x7
        )
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim * 7 * 7, 32 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_conv(x)
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
# Simple Latent UNet (no downsampling)
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


class SimpleLatentUNet(nn.Module):
    def __init__(self, latent_channels=8, time_emb_dim=64):
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
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_emb], dim=1)
        return self.net(x)


# ======================
# DDPM Scheduler (T=1000)
# ======================
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Will be moved to device in main
sqrt_alphas_cumprod = None
sqrt_one_minus_alphas_cumprod = None


def q_sample(z_0: Tensor, t: Tensor, noise: Tensor = None):
    if noise is None:
        noise = torch.randn_like(z_0)
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_prod * z_0 + sqrt_one_minus_alpha_prod * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    model.eval()
    b = shape[0]
    z = torch.randn(shape, device=device)
    
    for i in reversed(range(T)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        noise_pred = model(z, t)
        
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        
        if i == 0:
            noise = 0
        else:
            noise = torch.randn_like(z)
        
        coef1 = (1 - alpha_t) / (torch.sqrt(1 - alpha_cumprod_t))
        mean = (1 / torch.sqrt(alpha_t)) * (z - coef1 * noise_pred)
        variance = beta_t
        z = mean + torch.sqrt(variance) * noise
    return z


# ======================
# Training Functions
# ======================
def train_vae(vae, dataloader, epochs=20):
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    losses = []

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
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info(f"VAE Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
    
    torch.save(vae.state_dict(), model_dir / "vae.pth")
    # Save loss
    with open(model_dir / "vae_losses.json", "w") as f:
        json.dump(losses, f)
    logger.info("VAE saved.")
    return losses


def train_ldm(vae, unet, dataloader, epochs=50):
    vae.eval()
    unet.train()
    optimizer = torch.optim.Adam(unet.parameters(), lr=2e-4)
    losses = []

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
                mu, logvar = vae.encode(x)
                z_0 = vae.reparameterize(mu, logvar)  # ← 使用随机采样
            t = torch.randint(0, T, (x.size(0),), device=device).long()
            noise = torch.randn_like(z_0)
            z_noisy = q_sample(z_0, t, noise)
            noise_pred = unet(z_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info(f"LDM Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
    
    torch.save({
        'epoch': start_epoch + epochs,
        'model': unet.state_dict(),
        'optimizer': optimizer.state_dict()
    }, ldm_path)
    # Save loss
    with open(model_dir / "ldm_losses.json", "w") as f:
        json.dump(losses, f)
    logger.info("LDM saved.")
    return losses


def plot_losses(vae_losses, ldm_losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(vae_losses, label="VAE Loss")
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(ldm_losses, label="LDM Loss", color='orange')
    plt.title("LDM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(image_dir / "training_losses.png", dpi=150)
    plt.close()
    logger.info("Training loss curves saved.")


# ======================
# Test Functions
# ======================
def test_vae():
    vae = VAE(latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
    vae.eval()

    mode = input("VAE Test - (1) Random Generate (2) Reconstruction: ")
    if mode == "1":
        with torch.no_grad():
            z = torch.randn(8, 8, 7, 7, device=device)
            x_gen = vae.decode(z)
        x_gen = torch.clamp(x_gen, 0, 1).cpu()
        grid = torchvision.utils.make_grid(x_gen, nrow=4)
        save_path = image_dir / "vae_generated.png"
        plt.imsave(save_path, grid.permute(1, 2, 0).squeeze(), cmap="gray")
        logger.info(f"VAE generated saved: {save_path}")
    elif mode == "2":
        x, _ = next(iter(train_dataloader))
        x = x[:8].to(device)
        with torch.no_grad():
            recon, _, _ = vae(x)
        x, recon = x.cpu(), recon.cpu()
        fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        axes[0].imshow(torchvision.utils.make_grid(x, nrow=8)[0], cmap="gray")
        axes[1].imshow(torchvision.utils.make_grid(recon, nrow=8)[0], cmap="gray")
        for ax in axes: ax.axis("off")
        save_path = image_dir / "vae_recon.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"VAE recon saved: {save_path}")


def test_ldm():
    vae = VAE(latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
    vae.eval()

    unet = SimpleLatentUNet(latent_channels=8).to(device)
    ckpt = torch.load(model_dir / "ldm_unet.pth", map_location=device)
    unet.load_state_dict(ckpt['model'])
    unet.eval()

    mode = input("LDM Test - (1) Random Generate (2) Reconstruction: ")
    if mode == "1":
        z_T = torch.randn(8, 8, 7, 7, device=device)
        z_0 = p_sample_loop(unet, z_T.shape)
        with torch.no_grad():
            x_gen = vae.decode(z_0)
        x_gen = torch.clamp(x_gen, 0, 1).cpu()
        grid = torchvision.utils.make_grid(x_gen, nrow=4)
        save_path = image_dir / "ldm_generated.png"
        plt.imsave(save_path, grid.permute(1, 2, 0).squeeze(), cmap="gray")
        logger.info(f"LDM generated saved: {save_path}")
    elif mode == "2":
        x, _ = next(iter(train_dataloader))
        x = x[:8].to(device)
        with torch.no_grad():
            mu, logvar = vae.encode(x)
            z_0 = vae.reparameterize(mu, logvar)
            # Add noise at t=500
            t_val = 500
            t = torch.full((8,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(z_0)
            z_noisy = q_sample(z_0, t, noise)
            # Denoise from t=500
            z_recon = z_noisy
            for i in reversed(range(t_val)):
                t_step = torch.full((8,), i, device=device, dtype=torch.long)
                noise_pred = unet(z_recon, t_step)
                alpha_t = alphas[i]
                alpha_cumprod_t = alphas_cumprod[i]
                beta_t = betas[i]
                if i == 0:
                    noise_step = 0
                else:
                    noise_step = torch.randn_like(z_recon)
                coef1 = (1 - alpha_t) / (torch.sqrt(1 - alpha_cumprod_t))
                mean = (1 / torch.sqrt(alpha_t)) * (z_recon - coef1 * noise_pred)
                z_recon = mean + torch.sqrt(beta_t) * noise_step
            x_recon = vae.decode(z_recon)
        x, x_recon = x.cpu(), x_recon.cpu()
        fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        axes[0].imshow(torchvision.utils.make_grid(x, nrow=8)[0], cmap="gray")
        axes[1].imshow(torchvision.utils.make_grid(x_recon, nrow=8)[0], cmap="gray")
        for ax in axes: ax.axis("off")
        save_path = image_dir / "ldm_recon.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f"LDM recon saved: {save_path}")


# ======================
# Main
# ======================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["vae", "ldm", "test_vae", "test_ldm", "plot"], required=True)
    args = parser.parse_args()

    # Move scheduler tensors to device
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    if args.stage == "vae":
        vae = VAE(latent_dim=8).to(device)
        vae_losses = train_vae(vae, train_dataloader, epochs=20)
    elif args.stage == "ldm":
        vae = VAE(latent_dim=8).to(device)
        vae.load_state_dict(torch.load(model_dir / "vae.pth", map_location=device))
        unet = SimpleLatentUNet(latent_channels=8).to(device)
        ldm_losses = train_ldm(vae, unet, train_dataloader, epochs=50)
    elif args.stage == "test_vae":
        test_vae()
    elif args.stage == "test_ldm":
        test_ldm()
    elif args.stage == "plot":
        # Load losses and plot
        with open(model_dir / "vae_losses.json", "r") as f:
            vae_losses = json.load(f)
        with open(model_dir / "ldm_losses.json", "r") as f:
            ldm_losses = json.load(f)
        plot_losses(vae_losses, ldm_losses)