import os
import sys
from pathlib import Path
import time
import copy
import numpy as np
from datetime import datetime
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import math

# ---------- 路径配置 ----------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# VAE 权重路径（使用您训练好的模型）
VAE_MODEL_PATH = MODEL_DIR / "latest_model_convvae.pth"
# LDM 扩散模型保存路径
LDM_MODEL_PATH = MODEL_DIR / "ldm_diffusion.pth"
LDM_OPTIMIZER_PATH = MODEL_DIR / "ldm_optimizer.pth"
CLASSIFIER_PATH = MODEL_DIR / "mnist_classifier.pth"

# ---------- 导入自定义工具 ----------
project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
device = init_gpu_environment()
from utils.image_evaluator import evaluate_images
from utils.fid_evaluator import compute_fid

# ---------- 超参数 ----------
# VAE 相关
LATENT_DIM = 24                 # 与VAE训练时一致
BETA_MAX = 0.15
WARMUP_EPOCHS = 40
FREE_BITS_PER_DIM = 0.02
L1_WEIGHT = 0.15
EMA_DECAY = 0.999

# 扩散模型相关
BATCH_SIZE = 128                # 潜在空间训练可以稍大
NUM_EPOCHS = 100
LR = 1e-4
NUM_TIMESTEPS = 1000
INFERENCE_STEPS = 50            # 生成时推理步数

# 评估参数
N_GEN_EVAL = 10000
GEN_BATCH = 256
RECON_BATCH = 256

# ---------- 数据集 ----------
transform = transforms.Compose([transforms.ToTensor()])  # MNIST原始大小28x28
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 1. VAE 模型定义（与您的代码完全一致）
# =========================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class ConvVAE(nn.Module):
    def __init__(self, z_dim=24):
        super().__init__()
        self.z_dim = z_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28 -> 14
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),

            nn.Conv2d(32, 64, 4, 2, 1),  # 14 -> 7
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResBlock(64),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)

        # Decoder
        self.fc_dec = nn.Linear(z_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)    # 14 -> 28 (logits)
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -8.0, 6.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar

# =========================
# 2. 潜在空间扩散模型（MLP + 时间嵌入）
# =========================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.DEVICE
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LDMDiffusionModel(nn.Module):
    """潜在空间的扩散模型（MLP）"""
    def __init__(self, latent_dim=24, time_embed_dim=128, hidden_dims=[256, 512, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        layers = []
        input_dim = latent_dim + time_embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.SiLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, timestep):
        # x: [B, latent_dim]
        # timestep: [B] 整数
        t_emb = self.time_embed(timestep)          # [B, time_embed_dim]
        h = torch.cat([x, t_emb], dim=-1)          # [B, latent_dim+time_embed_dim]
        return self.net(h)

# =========================
# 3. 加载 VAE 权重并冻结
# =========================
def load_vae():
    vae = ConvVAE(z_dim=LATENT_DIM).to(device)
    if not VAE_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到VAE模型文件: {VAE_MODEL_PATH}，请先训练VAE。")
    checkpoint = torch.load(VAE_MODEL_PATH, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    # 冻结 VAE 参数
    for param in vae.parameters():
        param.requires_grad = False
    return vae

# =========================
# 4. 训练历史记录与检查点
# =========================
train_history = {
    'losses': [],
    'epoch': 0,
    'train_times': 0,
    'total_training_time': 0.0
}

def save_ldm_checkpoint(model, optimizer, epoch, loss_list, total_time, train_times):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': loss_list,
        'total_training_time': total_time,
        'train_times': train_times
    }, LDM_MODEL_PATH)

def load_ldm_checkpoint(model, optimizer):
    if not LDM_MODEL_PATH.exists():
        logger.info("第一次训练 LDM 模型")
        return 0, [], 0.0, 0
    try:
        ckpt = torch.load(LDM_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt.get('epoch', 0)
        losses = ckpt.get('losses', [])
        total_time = ckpt.get('total_training_time', 0.0)
        train_times = ckpt.get('train_times', 0)
        logger.info(f"已加载 LDM 模型, 起始 epoch={epoch}, 累计训练时长={total_time:.2f}s")
        return epoch, losses, total_time, train_times
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return 0, [], 0.0, 0

# =========================
# 5. 训练函数
# =========================
def train_ldm():
    # 加载 VAE
    vae = load_vae()
    # 初始化扩散模型和调度器
    diffusion_model = LDMDiffusionModel(latent_dim=LATENT_DIM).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    optimizer = optim.AdamW(diffusion_model.parameters(), lr=LR)

    start_epoch, loss_list, total_training_time, train_times = load_ldm_checkpoint(diffusion_model, optimizer)
    diffusion_model.train()
    end_epoch = start_epoch + NUM_EPOCHS
    start_time = time.time()

    for epoch in range(start_epoch, end_epoch):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch}")
        for step, (clean_images, _) in enumerate(progress_bar):
            clean_images = clean_images.to(device)
            bs = clean_images.size(0)

            # 通过 VAE 编码器得到潜在表示（使用重参数化采样）
            with torch.no_grad():
                mu, logvar = vae.encode(clean_images)
                z = vae.reparam(mu, logvar)           # [B, LATENT_DIM]

            # 添加噪声
            noise = torch.randn_like(z)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_z = noise_scheduler.add_noise(z, noise, timesteps)

            # 预测噪声
            noise_pred = diffusion_model(noisy_z, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / num_batches
        loss_list.append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{end_epoch}] Loss: {avg_loss:.4f}")

        # 每 10 个 epoch 生成样本预览
        if (epoch+1) % 10 == 0:
            generate_samples(vae, diffusion_model, noise_scheduler, epoch+1)

    elapsed = time.time() - start_time
    total_training_time += elapsed
    train_times += 1
    save_ldm_checkpoint(diffusion_model, optimizer, end_epoch, loss_list, total_training_time, train_times)
    logger.info(f"模型已保存，本次训练耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s")

    # 绘制损失曲线
    plt.figure(figsize=(8,5))
    plt.plot(loss_list)
    plt.title('LDM Training Loss')
    plt.xlabel('Epoch')
    plt.grid()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(MODEL_DIR / f"ldm_loss_{timestamp}.png", bbox_inches='tight')
    plt.close()
    logger.info("损失曲线已保存。")

# =========================
# 6. 采样函数（从噪声生成潜在向量）
# =========================
@torch.no_grad()
def sample_ldm(diffusion_model, noise_scheduler, num_samples, device,
               inference_steps=INFERENCE_STEPS, init_z=None):
    """
    从噪声（或给定初始向量）开始采样生成潜在向量。
    若 init_z 不为 None，则从中开始去噪。
    """
    diffusion_model.eval()
    if init_z is None:
        z_t = torch.randn(num_samples, LATENT_DIM, device=device)
    else:
        z_t = init_z.clone().to(device)

    timesteps = list(range(noise_scheduler.config.num_train_timesteps - 1, -1, -1))
    if inference_steps < NUM_TIMESTEPS:
        step_interval = NUM_TIMESTEPS // inference_steps
        timesteps = timesteps[::step_interval][:inference_steps]
    timesteps = sorted(timesteps, reverse=True)

    for t in timesteps:
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = diffusion_model(z_t, t_tensor)
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = noise_scheduler.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (z_t - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        if t > 0:
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_prod_t
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn_like(z_t)
            z_t = alpha_prod_t_prev.sqrt() * pred_original_sample + variance.sqrt() * noise
        else:
            z_t = pred_original_sample
    return z_t

def generate_samples(vae, diffusion_model, noise_scheduler, epoch=None):
    """生成图像并保存预览"""
    with torch.no_grad():
        z = sample_ldm(diffusion_model, noise_scheduler, 16, device, inference_steps=INFERENCE_STEPS)
        # 解码
        logits = vae.decode_logits(z)
        gen_imgs = torch.sigmoid(logits).cpu()
        # 保存
        grid = utils.make_grid(gen_imgs, nrow=4, normalize=True)
        if epoch is None:
            filename = IMAGE_DIR / f"ldm_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            filename = IMAGE_DIR / f"ldm_samples_epoch_{epoch}.png"
        utils.save_image(grid, filename)
        logger.info(f"样本图片已保存至 {filename}")

# =========================
# 7. 评估函数（复用分类器与FID）
# =========================
# 需要导入您现有的 mnist_classifier 和 diffusion_evaluator 或自行实现
# 这里假设您已经存在这些模块，若没有，可以复制原来的评估代码
# 为了简化，我们直接使用您VAE评估中已有的函数，但需要适配潜在空间生成
def evaluate_ldm():
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型，请先训练。")
        return

    # 加载 VAE 和扩散模型
    vae = load_vae()
    diffusion_model = LDMDiffusionModel(latent_dim=LATENT_DIM).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    ckpt = torch.load(LDM_MODEL_PATH, map_location=device, weights_only=False)
    diffusion_model.load_state_dict(ckpt["model_state_dict"])
    diffusion_model.eval()

    # ========== 随机生成评估 ==========
    logger.info("开始评估随机生成图像质量...")
    n_gen = N_GEN_EVAL
    batch_size = GEN_BATCH
    gen_imgs_list = []
    with torch.no_grad():
        remain = n_gen
        while remain > 0:
            bs = min(batch_size, remain)
            z = sample_ldm(diffusion_model, noise_scheduler, bs, device, inference_steps=INFERENCE_STEPS)
            logits = vae.decode_logits(z)
            imgs = torch.sigmoid(logits)
            gen_imgs_list.append(imgs.cpu())
            remain -= bs
    gen_imgs = torch.cat(gen_imgs_list, dim=0)

    # 分类器评估
    conf_gen, entropy_gen, coverage_gen = evaluate_images(gen_imgs, device=device)
    entropy_norm_gen = min(entropy_gen / 2.3026, 1.0)
    score_gen = 100.0 * (0.5 * conf_gen + 0.3 * coverage_gen + 0.2 * (1.0 - entropy_norm_gen))

    # 准备真实图像（用于 FID）
    real_imgs_list = []
    for x, _ in test_loader:
        real_imgs_list.append(x)
    real_imgs = torch.cat(real_imgs_list, dim=0)[:N_GEN_EVAL]

    # FID 评估
    fid_gen = compute_fid(gen_imgs, real_imgs, device=device)

    logger.info(f"随机生成 | 置信度: {conf_gen:.4f} | 熵: {entropy_gen:.4f} | 覆盖率: {coverage_gen:.4f} | 综合得分: {score_gen:.2f} | FID: {fid_gen:.2f}")

    # ========== 重构评估 ==========
    logger.info("开始评估重构图像质量（加噪再去噪）...")
    conf_rec, entropy_rec, coverage_rec, score_rec, fid_rec = evaluate_ldm_reconstruction(
        vae, diffusion_model, noise_scheduler, test_loader, device,
        n_samples=N_GEN_EVAL, batch_size=GEN_BATCH, noise_step=500
    )
    logger.info(f"重构图像 | 置信度: {conf_rec:.4f} | 熵: {entropy_rec:.4f} | 覆盖率: {coverage_rec:.4f} | 综合得分: {score_rec:.2f} | FID: {fid_rec:.2f}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = MODEL_DIR / f"ldm_eval_{timestamp}.txt"
    with open(result_file, 'w') as f:
        f.write("Random Generation:\n")
        f.write(f"  Confidence: {conf_gen:.4f}\n")
        f.write(f"  Entropy: {entropy_gen:.4f}\n")
        f.write(f"  Coverage: {coverage_gen:.4f}\n")
        f.write(f"  Final Score: {score_gen:.2f}\n")
        f.write(f"  FID: {fid_gen:.2f}\n\n")
        f.write("Reconstruction (Denoising):\n")
        f.write(f"  Confidence: {conf_rec:.4f}\n")
        f.write(f"  Entropy: {entropy_rec:.4f}\n")
        f.write(f"  Coverage: {coverage_rec:.4f}\n")
        f.write(f"  Final Score: {score_rec:.2f}\n")
        f.write(f"  FID: {fid_rec:.2f}\n")
    logger.info(f"评估结果已保存至 {result_file}")


def evaluate_ldm_reconstruction(vae, diffusion_model, noise_scheduler, test_loader, device,
                                n_samples=10000, batch_size=256, noise_step=500):
    """
    评估 LDM 重构图像的质量（加噪再降噪）。
    返回 (置信度, 熵, 覆盖率, FID)。
    """
    # 准备真实图像（用于 FID 和分类器评估）
    real_imgs_list = []
    for x, _ in test_loader:
        real_imgs_list.append(x)
    real_imgs = torch.cat(real_imgs_list, dim=0)[:n_samples]   # [n_samples,1,28,28]

    # 生成重构图像
    recon_imgs_list = []
    remain = n_samples
    with torch.no_grad():
        # 用真实图像通过 VAE 编码得到潜在向量
        for i in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - i)
            x = real_imgs[i:i+bs].to(device)
            mu, logvar = vae.encode(x)
            # 使用重参数化采样得到 z
            z = vae.reparam(mu, logvar)          # [bs, latent_dim]
            # 对 z 加噪到指定时间步 noise_step
            noise = torch.randn_like(z)
            timesteps = torch.full((bs,), noise_step, device=device, dtype=torch.long)
            noisy_z = noise_scheduler.add_noise(z, noise, timesteps)
            # 去噪（从 noisy_z 开始）
            denoised_z = sample_ldm(diffusion_model, noise_scheduler, bs, device,
                                    inference_steps=INFERENCE_STEPS, init_z=noisy_z)
            # 解码得到重构图像
            logits = vae.decode_logits(denoised_z)
            recon_imgs = torch.sigmoid(logits)   # [bs,1,28,28]
            recon_imgs_list.append(recon_imgs.cpu())
    recon_imgs = torch.cat(recon_imgs_list, dim=0)[:n_samples]

    # 分类器评估
    conf, entropy, coverage = evaluate_images(recon_imgs, device=device)
    entropy_norm = min(entropy / 2.3026, 1.0)
    score = 100.0 * (0.5 * conf + 0.3 * coverage + 0.2 * (1.0 - entropy_norm))

    # FID 评估（真实图像和重构图像）
    fid = compute_fid(recon_imgs, real_imgs, device=device)

    return conf, entropy, coverage, score, fid

# =========================
# 8. 生成模式（供用户调用）
# =========================
def generate_mode():
    """从LDM生成9张图像"""
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型。")
        return
    vae = load_vae()
    diffusion_model = LDMDiffusionModel(latent_dim=LATENT_DIM).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    ckpt = torch.load(LDM_MODEL_PATH, map_location=device, weights_only=False)
    diffusion_model.load_state_dict(ckpt["model_state_dict"])
    diffusion_model.eval()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with torch.no_grad():
        z = sample_ldm(diffusion_model, noise_scheduler, 9, device, inference_steps=INFERENCE_STEPS)
        logits = vae.decode_logits(z)
        imgs = torch.sigmoid(logits).cpu().squeeze(1)  # [9,28,28]
        # 保存单张大图
        for i in range(9):
            fig, ax = plt.subplots(figsize=(5,5), dpi=100)
            ax.imshow(imgs[i].numpy(), cmap='gray')
            ax.axis('off')
            plt.savefig(IMAGE_DIR / f"ldm_generate_{timestamp}_{i+1:02d}.png", bbox_inches='tight')
            plt.close()
        # 宫格图
        fig, axes = plt.subplots(3, 3, figsize=(12,12))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i].numpy(), cmap='gray')
            ax.axis('off')
        plt.savefig(IMAGE_DIR / f"ldm_generate_{timestamp}_grid.png", bbox_inches='tight')
        plt.close()
        logger.info("生成模式完成，图片已保存。")

def reconstruct_mode():
    """从测试集取 9 张图像，加噪后去噪，并保存对比图"""
    if not LDM_MODEL_PATH.exists():
        logger.error("未找到 LDM 模型。")
        return

    vae = load_vae()
    diffusion_model = LDMDiffusionModel(latent_dim=LATENT_DIM).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    ckpt = torch.load(LDM_MODEL_PATH, map_location=device, weights_only=False)
    diffusion_model.load_state_dict(ckpt["model_state_dict"])
    diffusion_model.eval()

    # 取 9 张测试图像
    real_imgs, _ = next(iter(test_loader))
    real_imgs = real_imgs[:9].to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with torch.no_grad():
        # 编码
        mu, logvar = vae.encode(real_imgs)
        z = vae.reparam(mu, logvar)
        # 加噪到 t=500
        noise = torch.randn_like(z)
        timesteps = torch.full((9,), 500, device=device, dtype=torch.long)
        noisy_z = noise_scheduler.add_noise(z, noise, timesteps)
        # 去噪
        denoised_z = sample_ldm(diffusion_model, noise_scheduler, 9, device,
                                inference_steps=INFERENCE_STEPS, init_z=noisy_z)
        # 解码
        logits = vae.decode_logits(denoised_z)
        recon_imgs = torch.sigmoid(logits)   # [9,1,28,28]

        # 保存对比图
        real_np = real_imgs.cpu().squeeze().numpy()
        recon_np = recon_imgs.cpu().squeeze().numpy()
        for i in range(9):
            combined = np.hstack((real_np[i], recon_np[i]))
            plt.figure(figsize=(10,5))
            plt.imshow(combined, cmap='gray')
            plt.axis('off')
            plt.savefig(IMAGE_DIR / f"ldm_recon_{timestamp}_{i+1:02d}.png", bbox_inches='tight')
            plt.close()
        # 宫格对比
        fig, axes = plt.subplots(3, 3, figsize=(15,15))
        for i, ax in enumerate(axes.flat):
            combined = np.hstack((real_np[i], recon_np[i]))
            ax.imshow(combined, cmap='gray')
            ax.axis('off')
        plt.savefig(IMAGE_DIR / f"ldm_recon_{timestamp}_grid.png", bbox_inches='tight')
        plt.close()
        logger.info("重构模式完成，图片已保存。")

def show_model_status():
    """显示 LDM 模型状态"""
    if not LDM_MODEL_PATH.exists():
        logger.info("尚未发现 LDM 模型文件。")
        return
    try:
        ckpt = torch.load(LDM_MODEL_PATH, map_location=device)
        epoch = ckpt.get('epoch', 0)
        losses = ckpt.get('losses', [])
        latest_loss = losses[-1] if losses else "N/A"
        train_times = ckpt.get('train_times', 0)
        total_time = ckpt.get('total_training_time', 0.0)

        logger.info("="*40)
        logger.info("     LDM 模型状态报告")
        logger.info(f"  已训练总轮数:      {epoch}")
        logger.info(f"  累计训练次数:      {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f"  最近平均 Loss:     {latest_loss:.4f}")
        if total_time < 3600:
            logger.info(f"  训练时长：{total_time/60:.1f} 分钟")
        else:
            logger.info(f"  训练时长：{total_time/3600:.1f} 小时")
        logger.info("="*40)
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")

# =========================
# 9. 主菜单
# =========================
def main():
    while True:
        logger.info("\n" + "="*40)
        logger.info("    潜在扩散模型管理系统 (LDM)")
        logger.info(" [1] 训练 LDM 模型")
        logger.info(" [2] 测试模式 (Generate/Reconstruct)")
        logger.info(" [3] 查看模型状态")
        logger.info(" [4] 评估模型")
        logger.info(" [0/exit] 退出程序")
        logger.info("="*40)
        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == '1':
            train_ldm()
        elif choice == '2':
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (从先验采样生成)")
                logger.info("  [2] Reconstruct (加噪后去噪对比)")
                logger.info("  [0/exit] 返回主菜单并完全退出")
                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == '1':
                    generate_mode()
                elif sub_choice == '2':
                    reconstruct_mode()
                elif sub_choice in ['0', 'exit']:
                    logger.info("退出程序...")
                    sys.exit(0)
                else:
                    logger.info("  无效输入，请输入 1, 2 或 0")
        elif choice == '3':
            show_model_status()
        elif choice == '4':
            evaluate_ldm()
        elif choice in ['0', 'exit']:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3, 4 或 0")

if __name__ == '__main__':
    main()