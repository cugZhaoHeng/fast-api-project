# -*- coding: utf-8 -*-
"""
MNIST Conv-VAE (Improved, Full Runnable Version)
改进目标：
1) 提升生成可识别度（减少“中间认不出来”）
2) 训练更稳定、时间更长
3) 同时提供两种采样可视化：
   - 标准正态采样 N(0,I)
   - 聚合后验采样（更清晰，推荐）
4) 保留完整评估：
   ELBO/Recon/KL + 分类器生成指标 + Final Score + FID(ResNet18)

依赖：
pip install torch torchvision matplotlib scipy numpy
"""

import warnings
warnings.filterwarnings("ignore")

import copy
import random
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# =========================
# Config
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 120                 # 训练时间更长
LR = 1e-3
WEIGHT_DECAY = 1e-5
LATENT_DIM = 24

BETA_MAX = 0.15              # 降低KL压力，避免过分“平均化”
WARMUP_EPOCHS = 40           # 更长warmup
FREE_BITS_PER_DIM = 0.02     # KL free bits
L1_WEIGHT = 0.15             # 辅助边缘清晰（不要太大，防止失真）
EMA_DECAY = 0.999

NUM_WORKERS = 0              # Windows稳定优先
PIN_MEMORY = torch.cuda.is_available()

N_GEN_EVAL = 10000
GEN_BATCH = 256


# =========================
# Data
# =========================
transform = transforms.ToTensor()

train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)


# =========================
# Model
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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


# =========================
# Loss & Schedules
# =========================
def beta_schedule(epoch):
    if epoch >= WARMUP_EPOCHS:
        return BETA_MAX
    return BETA_MAX * (epoch / max(1, WARMUP_EPOCHS))


def loss_fn(logits, x, mu, logvar, beta=1.0, free_bits_per_dim=0.0, l1_weight=0.0):
    # BCE recon per sample
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none').flatten(1).sum(1)
    # L1 recon per sample
    x_hat = torch.sigmoid(logits)
    l1 = F.l1_loss(x_hat, x, reduction='none').flatten(1).sum(1)

    # KL per dim -> free bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    if free_bits_per_dim > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits_per_dim)
    kl = kl_per_dim.sum(1)  # [B]

    recon = bce + l1_weight * l1
    elbo = recon + beta * kl
    loss = elbo.mean()
    return loss, recon.sum(), kl.sum()


# =========================
# Training
# =========================
model = ConvVAE(z_dim=LATENT_DIM).to(device)
ema = EMA(model, decay=EMA_DECAY)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("Device:", device)
print("Start Training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    beta = beta_schedule(epoch)
    total_loss, total_n = 0.0, 0

    for x, _ in train_loader:
        x = x.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad(set_to_none=True)
        logits, mu, logvar = model(x)
        loss, _, _ = loss_fn(
            logits, x, mu, logvar,
            beta=beta,
            free_bits_per_dim=FREE_BITS_PER_DIM,
            l1_weight=L1_WEIGHT
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)

        total_loss += loss.item() * bs
        total_n += bs

    print(f"Epoch {epoch:03d} | beta={beta:.3f} | train loss/sample={total_loss / total_n:.3f}")


# =========================
# Visualization
# =========================
@torch.no_grad()
def show_samples_standard_normal(model_eval, latent_dim, n=64, title="Samples from N(0,I)"):
    model_eval.eval()
    z = torch.randn(n, latent_dim, device=device)
    gen = torch.sigmoid(model_eval.decode_logits(z)).cpu()

    plt.figure(figsize=(8, 8))
    side = int(n ** 0.5)
    for i in range(n):
        plt.subplot(side, side, i + 1)
        plt.imshow(gen[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_samples_agg_posterior(model_eval, train_loader, n=64, title="Samples from Aggregated Posterior"):
    model_eval.eval()
    zs = []
    need = n
    it = iter(train_loader)

    while need > 0:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, _ = next(it)
        x = x.to(device, non_blocking=True)

        mu, logvar = model_eval.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        take = min(need, z.size(0))
        zs.append(z[:take])
        need -= take

    z_all = torch.cat(zs, dim=0)
    gen = torch.sigmoid(model_eval.decode_logits(z_all)).cpu()

    plt.figure(figsize=(8, 8))
    side = int(n ** 0.5)
    for i in range(n):
        plt.subplot(side, side, i + 1)
        plt.imshow(gen[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


eval_model = ema.shadow
show_samples_standard_normal(eval_model, LATENT_DIM, n=64, title="Generated MNIST (N(0,I))")
show_samples_agg_posterior(eval_model, train_loader, n=64, title="Generated MNIST (Aggregated Posterior, clearer)")


# =========================
# Evaluation Core
# =========================
@torch.no_grad()
def evaluate_vae_on_test(model_eval, loader, beta=1.0):
    model_eval.eval()
    total_loss, total_recon, total_kl, total_n = 0.0, 0.0, 0.0, 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits, mu, logvar = model_eval(x)
        loss, recon, kl = loss_fn(
            logits, x, mu, logvar,
            beta=beta,
            free_bits_per_dim=0.0,
            l1_weight=L1_WEIGHT
        )
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon.item()
        total_kl += kl.item()
        total_n += bs

    return total_loss / total_n, total_recon / total_n, total_kl / total_n


@torch.no_grad()
def evaluate_test_recon_only(model_eval, loader):
    model_eval.eval()
    total_recon, total_n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits, _, _ = model_eval(x)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
        total_recon += recon.item()
        total_n += x.size(0)
    return total_recon / total_n


# =========================
# Classifier for generation metrics
# =========================
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_mnist_classifier(train_loader, test_loader, epochs=6, lr=1e-3):
    clf = MNISTClassifier().to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = ce(clf(x), y)
            loss.backward()
            opt.step()

    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = clf(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return clf, correct / total


@torch.no_grad()
def evaluate_generated_images(model_eval, clf, latent_dim, n_gen=10000, batch_size=256, mode="normal", ref_loader=None):
    """
    mode:
      - "normal": z ~ N(0, I)
      - "agg": z ~ aggregated posterior
    """
    model_eval.eval()
    clf.eval()

    all_probs = []
    remain = n_gen
    ref_iter = iter(ref_loader) if ref_loader is not None else None

    while remain > 0:
        bs = min(batch_size, remain)

        if mode == "normal":
            z = torch.randn(bs, latent_dim, device=device)

        elif mode == "agg":
            if ref_iter is None:
                raise ValueError("mode='agg' requires ref_loader")
            xs = []
            need = bs
            while need > 0:
                try:
                    x, _ = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_loader)
                    x, _ = next(ref_iter)
                take = min(need, x.size(0))
                xs.append(x[:take])
                need -= take
            x_ref = torch.cat(xs, dim=0).to(device, non_blocking=True)
            mu, logvar = model_eval.encode(x_ref)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            raise ValueError("mode must be 'normal' or 'agg'")

        gen = torch.sigmoid(model_eval.decode_logits(z))
        probs = torch.softmax(clf(gen), dim=1)
        all_probs.append(probs.cpu())
        remain -= bs

    probs = torch.cat(all_probs, dim=0)

    max_conf = probs.max(dim=1).values.mean().item()

    eps = 1e-8
    ent_per_img = -(probs * (probs + eps).log()).sum(dim=1)
    mean_entropy = ent_per_img.mean().item()

    p_y = probs.mean(dim=0)
    uniform = torch.full_like(p_y, 0.1)
    l1_dist = torch.abs(p_y - uniform).sum().item()
    coverage = 1.0 - l1_dist / 1.8

    return max_conf, mean_entropy, coverage


# =========================
# FID (ResNet18 feature)
# =========================
def build_resnet18_feature_extractor():
    try:
        feat_net = resnet18(weights=ResNet18_Weights.DEFAULT)
        mode = "pretrained"
    except Exception:
        feat_net = resnet18(weights=None)
        mode = "random_init"
        print("[Warning] ResNet18 pretrained weights unavailable, fallback random. FID参考性下降。")
    feat_net.fc = nn.Identity()
    feat_net = feat_net.to(device).eval()
    return feat_net, mode


@torch.no_grad()
def _extract_resnet18_features(feat_net, x):
    x = x.repeat(1, 3, 1, 1)
    x = TF.resize(x, [224, 224], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    feats = feat_net(x)
    return feats.cpu().numpy()


def _compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = linalg.sqrtm((sigma1 @ sigma2).astype(np.float64), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


@torch.no_grad()
def evaluate_fid_resnet18(model_eval, test_loader, latent_dim, n_gen=10000, batch_size=256, mode="normal", ref_loader=None):
    model_eval.eval()
    feat_net, fid_mode = build_resnet18_feature_extractor()

    # real feats
    real_feats = []
    for x, _ in test_loader:
        x = x.to(device, non_blocking=True)
        real_feats.append(_extract_resnet18_features(feat_net, x))
    real_feats = np.concatenate(real_feats, axis=0)

    # fake feats
    fake_feats = []
    remain = n_gen
    ref_iter = iter(ref_loader) if ref_loader is not None else None

    while remain > 0:
        bs = min(batch_size, remain)

        if mode == "normal":
            z = torch.randn(bs, latent_dim, device=device)
        elif mode == "agg":
            if ref_iter is None:
                raise ValueError("mode='agg' requires ref_loader")
            xs = []
            need = bs
            while need > 0:
                try:
                    x_ref, _ = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_loader)
                    x_ref, _ = next(ref_iter)
                take = min(need, x_ref.size(0))
                xs.append(x_ref[:take])
                need -= take
            x_ref = torch.cat(xs, dim=0).to(device, non_blocking=True)
            mu, logvar = model_eval.encode(x_ref)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            raise ValueError("mode must be 'normal' or 'agg'")

        gen = torch.sigmoid(model_eval.decode_logits(z))
        fake_feats.append(_extract_resnet18_features(feat_net, gen))
        remain -= bs

    fake_feats = np.concatenate(fake_feats, axis=0)

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    fid = _compute_fid(mu_r, sigma_r, mu_f, sigma_f)
    return fid, fid_mode


# =========================
# Final Evaluation
# =========================
test_loss, test_recon, test_kl = evaluate_vae_on_test(eval_model, test_loader, beta=BETA_MAX)
test_recon_only = evaluate_test_recon_only(eval_model, test_loader)

clf, clf_test_acc = train_mnist_classifier(train_loader, test_loader, epochs=6, lr=1e-3)

# 1) normal prior sampling metrics
gen_conf_n, gen_entropy_n, gen_coverage_n = evaluate_generated_images(
    eval_model, clf, latent_dim=LATENT_DIM, n_gen=N_GEN_EVAL, batch_size=GEN_BATCH, mode="normal"
)
entropy_norm_n = min(gen_entropy_n / 2.3026, 1.0)
final_score_n = 100.0 * (0.5 * gen_conf_n + 0.3 * gen_coverage_n + 0.2 * (1.0 - entropy_norm_n))
fid_n, fid_mode_n = evaluate_fid_resnet18(
    eval_model, test_loader, latent_dim=LATENT_DIM, n_gen=N_GEN_EVAL, batch_size=GEN_BATCH, mode="normal"
)

# 2) aggregated posterior sampling metrics
gen_conf_a, gen_entropy_a, gen_coverage_a = evaluate_generated_images(
    eval_model, clf, latent_dim=LATENT_DIM, n_gen=N_GEN_EVAL, batch_size=GEN_BATCH, mode="agg", ref_loader=train_loader
)
entropy_norm_a = min(gen_entropy_a / 2.3026, 1.0)
final_score_a = 100.0 * (0.5 * gen_conf_a + 0.3 * gen_coverage_a + 0.2 * (1.0 - entropy_norm_a))
fid_a, fid_mode_a = evaluate_fid_resnet18(
    eval_model, test_loader, latent_dim=LATENT_DIM, n_gen=N_GEN_EVAL, batch_size=GEN_BATCH, mode="agg", ref_loader=train_loader
)

print("\n========== Evaluation Results ==========")
print(f"Test Loss (ELBO approx): {test_loss:.4f}      (越小越好)")
print(f"Test Recon BCE+L1:       {test_recon:.4f}      (越小越好)")
print(f"Test KL:                 {test_kl:.4f}      (需结合Recon看)")
print(f"Classifier Test Acc:     {clf_test_acc:.4f}      (越大越好)")
print(f"Test Recon Loss only:    {test_recon_only:.4f}   (越小越好)")

print("\n---- Sampling by N(0,I) ----")
print(f"Gen Confidence:          {gen_conf_n:.4f}      (越大越好)")
print(f"Gen Entropy:             {gen_entropy_n:.4f}      (越小越好)")
print(f"Gen Coverage:            {gen_coverage_n:.4f}      (越大越好)")
print(f"Final Score:             {final_score_n:.2f}/100  (越大越好)")
print(f"FID (ResNet18-{fid_mode_n}): {fid_n:.4f}      (越小越好)")

print("\n---- Sampling by Aggregated Posterior (推荐) ----")
print(f"Gen Confidence:          {gen_conf_a:.4f}      (越大越好)")
print(f"Gen Entropy:             {gen_entropy_a:.4f}      (越小越好)")
print(f"Gen Coverage:            {gen_coverage_a:.4f}      (越大越好)")
print(f"Final Score:             {final_score_a:.2f}/100  (越大越好)")
print(f"FID (ResNet18-{fid_mode_a}): {fid_a:.4f}      (越小越好)")