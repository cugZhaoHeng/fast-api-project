# -*- coding: utf-8 -*-
"""
Simple Conv-VAE for MNIST (精简版)
目标：用最简单训练流程验证普通VAE效果上限
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ====== config ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
LATENT_DIM = 32
BETA_MAX = 1.0
WARMUP_EPOCHS = 10

# ====== data ======
transform = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ====== model ======
class ConvVAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),  # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 7x7
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)

        self.fc_dec = nn.Linear(z_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 28x28 logits
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

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


def loss_fn(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def beta_schedule(epoch):
    if epoch >= WARMUP_EPOCHS:
        return BETA_MAX
    return BETA_MAX * (epoch / WARMUP_EPOCHS)


# ====== train ======
model = ConvVAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    beta = beta_schedule(epoch)
    train_loss = 0

    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(x)
        loss, _, _ = loss_fn(logits, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch:02d} | beta={beta:.3f} | train loss per sample={train_loss / len(train_ds):.3f}")

# ====== sample ======
model.eval()
with torch.no_grad():
    z = torch.randn(64, LATENT_DIM, device=device)
    gen = torch.sigmoid(model.decode_logits(z)).cpu()

plt.figure(figsize=(8, 8))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(gen[i, 0], cmap='gray')
    plt.axis('off')
plt.suptitle("Generated MNIST by Simple Conv-VAE")
plt.tight_layout()
plt.show()

# ====== evaluation ======
# 说明：
# 1) VAE本体指标
#    - Test Loss (ELBO近似): 越小越好
#    - Test Recon BCE:       越小越好
#    - Test KL:              需结合看（不是绝对越小越好）
# 2) 生成质量指标（借助MNIST分类器）
#    - Gen Confidence:       越大越好（生成图像被分类器确信是某个数字）
#    - Gen Entropy:          越小越好（单张图像类别分布越“明确”）
#    - Gen Coverage:         越大越好（10个数字类别覆盖更均衡）
# 3) final_score（0~100）：越大越好（综合分）

@torch.no_grad()
def evaluate_vae_on_test(model, loader, beta=1.0):
    model.eval()
    total_loss, total_recon, total_kl, total_n = 0.0, 0.0, 0.0, 0

    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        loss, recon, kl = loss_fn(logits, x, mu, logvar, beta=beta)

        bs = x.size(0)
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_n += bs

    # 注意：loss_fn是sum over batch+pixels+latent，这里按样本平均
    return (
        total_loss / total_n,   # per-sample
        total_recon / total_n,  # per-sample
        total_kl / total_n      # per-sample
    )


# 一个轻量MNIST分类器，用于评估“生成图像是否像数字”
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_mnist_classifier(train_loader, test_loader, epochs=3, lr=1e-3):
    clf = MNISTClassifier().to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = clf(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

    # 简单看下分类器准确率
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = clf(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    return clf, acc


@torch.no_grad()
def evaluate_generated_images(model, clf, n_gen=10000, batch_size=256):
    model.eval()
    clf.eval()

    all_probs = []
    remain = n_gen
    while remain > 0:
        bs = min(batch_size, remain)
        z = torch.randn(bs, LATENT_DIM, device=device)
        gen = torch.sigmoid(model.decode_logits(z))  # [bs,1,28,28] in [0,1]
        logits = clf(gen)
        probs = torch.softmax(logits, dim=1)         # [bs,10]
        all_probs.append(probs.cpu())
        remain -= bs

    probs = torch.cat(all_probs, dim=0)  # [n_gen,10]

    # 1) 平均最大类概率（越大越好）
    max_conf = probs.max(dim=1).values.mean().item()

    # 2) 单图熵（越小越好，表示更“像一个明确数字”）
    eps = 1e-8
    ent_per_img = -(probs * (probs + eps).log()).sum(dim=1)
    mean_entropy = ent_per_img.mean().item()

    # 3) 类别覆盖度（越大越好，越接近1表示10类更均衡）
    p_y = probs.mean(dim=0)  # [10]
    uniform = torch.full_like(p_y, 1.0 / 10.0)
    l1_dist = torch.abs(p_y - uniform).sum().item()   # [0, 1.8]
    coverage = 1.0 - l1_dist / 1.8                    # 归一化到[0,1]，越大越好

    return max_conf, mean_entropy, coverage


# ---- 运行评估 ----
test_loss, test_recon, test_kl = evaluate_vae_on_test(model, test_loader, beta=1.0)

clf, clf_test_acc = train_mnist_classifier(train_loader, test_loader, epochs=3, lr=1e-3)
gen_conf, gen_entropy, gen_coverage = evaluate_generated_images(model, clf, n_gen=10000, batch_size=256)

# 综合分（0~100，越大越好）
# 设计思路：高置信度 + 高覆盖度 + 低熵
# 其中熵最大约 ln(10)=2.3026，做归一化后取(1-归一化熵)
entropy_norm = min(gen_entropy / 2.3026, 1.0)
final_score = 100.0 * (0.5 * gen_conf + 0.3 * gen_coverage + 0.2 * (1.0 - entropy_norm))

# ====== add-on: FID + explicit test recon loss print ======
# 依赖: scipy
# pip install scipy

import numpy as np
from scipy import linalg
from torchvision.models import resnet18
import torchvision.transforms.functional as TF

@torch.no_grad()
def evaluate_test_recon_only(model, loader):
    """单独计算 Test Recon BCE（per-sample）"""
    model.eval()
    total_recon, total_n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
        total_recon += recon.item()
        total_n += x.size(0)
    return total_recon / total_n


@torch.no_grad()
def _extract_resnet18_features(feat_net, x):
    """
    x: [B,1,28,28], range [0,1]
    return: numpy [B,512]
    """
    x = x.repeat(1, 3, 1, 1)                       # -> [B,3,28,28]
    x = TF.resize(x, [224, 224], antialias=True)   # -> [B,3,224,224]

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    x = (x - mean) / std

    feats = feat_net(x)  # [B,512]
    return feats.cpu().numpy()


def _compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = linalg.sqrtm((sigma1 @ sigma2).astype(np.float64), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


@torch.no_grad()
def evaluate_fid_resnet18(model, test_loader, latent_dim, n_gen=10000, batch_size=256):
    model.eval()

    # 仅作特征提取，不训练
    feat_net = resnet18(weights=None)  # 若环境支持预训练可替换成预训练权重
    feat_net.fc = nn.Identity()
    feat_net = feat_net.to(device).eval()

    # real features
    real_feats = []
    for x, _ in test_loader:
        x = x.to(device)
        real_feats.append(_extract_resnet18_features(feat_net, x))
    real_feats = np.concatenate(real_feats, axis=0)

    # fake features
    fake_feats = []
    remain = n_gen
    while remain > 0:
        bs = min(batch_size, remain)
        z = torch.randn(bs, latent_dim, device=device)
        gen = torch.sigmoid(model.decode_logits(z))
        fake_feats.append(_extract_resnet18_features(feat_net, gen))
        remain -= bs
    fake_feats = np.concatenate(fake_feats, axis=0)

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    return _compute_fid(mu_r, sigma_r, mu_f, sigma_f)


# ---- run extra metrics ----
test_recon_only = evaluate_test_recon_only(model, test_loader)  # 越小越好
fid_score = evaluate_fid_resnet18(model, test_loader, LATENT_DIM, n_gen=10000, batch_size=256)  # 越小越好


print("\n========== Evaluation Results ==========")
print(f"Test Loss (ELBO approx): {test_loss:.4f}      (越小越好)")
print(f"Test Recon BCE:          {test_recon:.4f}      (越小越好)")
print(f"Test KL:                 {test_kl:.4f}      (需结合Recon看)")

print(f"Classifier Test Acc:     {clf_test_acc:.4f}      (越大越好，分类器可靠性参考)")
print(f"Gen Confidence:          {gen_conf:.4f}      (越大越好)")
print(f"Gen Entropy:             {gen_entropy:.4f}      (越小越好)")
print(f"Gen Coverage:            {gen_coverage:.4f}      (越大越好)")

print(f"\nFinal Score:             {final_score:.2f}/100  (越大越好)")
print(f"Test Recon Loss (BCE per sample): {test_recon_only:.4f}  (越小越好)")
print(f"FID (ResNet18 feature):           {fid_score:.4f}  (越小越好, 0最好)")