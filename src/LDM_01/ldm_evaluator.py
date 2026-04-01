import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
from scipy import linalg
from pathlib import Path

# ---------- 路径配置（根据实际项目调整） ----------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent.parent
MODEL_DIR = CURRENT_DIR / 'models'
CLASSIFIER_PATH = MODEL_DIR / "mnist_classifier.pth"

# ---------- 分类器加载 ----------
def load_classifier(device):
    """
    加载预先训练好的 MNIST 分类器。
    这里假设分类器模型位于 CLASSIFIER_PATH，且是一个接受 (batch,1,28,28) 输入、输出 10 类 logits 的模型。
    如果没有，可以自行训练或使用简单的 CNN。
    """
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"分类器模型未找到: {CLASSIFIER_PATH}，请先训练或下载。")
    try:
        # 这里需要根据您的分类器定义来导入，例如：
        from mnist_classifier import SimpleCNN  # 假设分类器类名为 SimpleCNN
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        model.eval()
        return model
    except ImportError:
        # 如果无法导入，提供一个简单的 CNN 定义（与训练时一致）
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)
                self.pool = nn.MaxPool2d(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        model.eval()
        return model

# ---------- FID 相关 ----------
def build_resnet18_feature_extractor(device):
    """构建 ResNet18 特征提取器（移除分类头）"""
    feat_net = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    feat_net.fc = nn.Identity()
    feat_net.eval()
    return feat_net

def extract_features(feat_net, images, device):
    """
    images: (B, 1, 28, 28) 张量，范围 [0,1]
    返回 (B, 512) 特征向量
    """
    # 转换为 3 通道并 resize 到 224x224
    images = images.repeat(1, 3, 1, 1)  # 复制到 3 通道
    images = TF.resize(images, [224, 224], antialias=True)
    # 标准化到 ImageNet 均值标准差
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    images = (images - mean) / std
    with torch.no_grad():
        feats = feat_net(images)
    return feats.cpu().numpy()

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算 FID 分数"""
    covmean, _ = linalg.sqrtm((sigma1 @ sigma2).astype(np.float64), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def fid_generation(vae, diffusion_model, noise_scheduler, test_loader, device, n_gen=10000, batch_size=256):
    """计算生成图像的 FID"""
    feat_net = build_resnet18_feature_extractor(device)

    # 真实图像特征
    real_feats = []
    for x, _ in test_loader:
        x = x.to(device)  # [B,1,28,28]
        real_feats.append(extract_features(feat_net, x, device))
    real_feats = np.concatenate(real_feats, axis=0)
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)

    # 生成图像特征
    fake_feats = []
    remain = n_gen
    while remain > 0:
        bs = min(batch_size, remain)
        # 使用 LDM 采样生成潜在向量
        z = sample_ldm(diffusion_model, noise_scheduler, bs, device)  # 需要实现采样函数
        with torch.no_grad():
            logits = vae.decode_logits(z)
            imgs = torch.sigmoid(logits)  # [bs,1,28,28]
        fake_feats.append(extract_features(feat_net, imgs, device))
        remain -= bs
    fake_feats = np.concatenate(fake_feats, axis=0)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    fid = compute_fid(mu_r, sigma_r, mu_f, sigma_f)
    return fid

# ---------- 分类器评估 ----------
def classifier_evaluation(vae, diffusion_model, noise_scheduler, device, n_gen=10000, batch_size=256):
    """
    使用分类器评估生成图像的质量，返回 (置信度, 熵, 覆盖率, 综合得分)
    """
    classifier = load_classifier(device)
    classifier.eval()

    all_preds = []
    all_probs = []
    remain = n_gen
    while remain > 0:
        bs = min(batch_size, remain)
        z = sample_ldm(diffusion_model, noise_scheduler, bs, device)
        with torch.no_grad():
            logits = vae.decode_logits(z)
            imgs = torch.sigmoid(logits)
            outputs = classifier(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        remain -= bs

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)

    conf = all_probs.max(dim=1)[0].mean().item()
    entropy = - (all_probs * torch.log(all_probs + 1e-8)).sum(dim=1).mean().item()
    coverage = len(torch.unique(all_preds)) / 10.0
    # 综合得分（示例，可根据需要调整）
    score = 100.0 * (0.5 * conf + 0.3 * coverage + 0.2 * (1.0 - min(entropy / 2.3026, 1.0)))
    return conf, entropy, coverage, score

# ---------- 需要从 LDM 主文件导入的采样函数 ----------
# 为了避免循环依赖，这里只声明函数原型，实际在调用时由外部传入
def sample_ldm(diffusion_model, noise_scheduler, num_samples, device, inference_steps=50):
    """
    这个函数应该在 LDM 主文件中定义，并传入 evaluator 使用。
    如果评估模块独立，可以在这里实现一个简单的版本（但会重复代码）。
    我们推荐在主文件中定义，然后作为参数传递。
    """
    # 注意：这个函数仅作占位，实际使用时需要从主文件导入或传入。
    raise NotImplementedError("请从 LDM 主文件导入 sample_ldm 函数，或将其作为参数传递给评估函数。")

# ---------- 统一评估接口 ----------
def evaluate_ldm(vae, diffusion_model, noise_scheduler, test_loader, device,
                 n_gen=10000, batch_size=256, inference_steps=50):
    """
    执行 LDM 的完整评估，返回字典包含各项指标。
    需要传入 sample_ldm 函数，因为该函数依赖于模型和调度器。
    """
    # 使用传入的采样函数（闭包）
    def _sample_ldm(num_samples):
        return sample_ldm(diffusion_model, noise_scheduler, num_samples, device, inference_steps)

    # 临时将 sample_ldm 绑定到本模块的全局，方便内部函数调用
    import sys
    sys.modules[__name__].sample_ldm = _sample_ldm

    # 分类器评估
    conf, entropy, coverage, score = classifier_evaluation(vae, diffusion_model, noise_scheduler, device, n_gen, batch_size)
    # FID 评估
    fid = fid_generation(vae, diffusion_model, noise_scheduler, test_loader, device, n_gen, batch_size)

    return {
        'confidence': conf,
        'entropy': entropy,
        'coverage': coverage,
        'score': score,
        'fid': fid
    }

if __name__ == "__main__":
    pass