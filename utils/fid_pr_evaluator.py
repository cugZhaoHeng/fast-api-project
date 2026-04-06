"""
生成流形评估模块（基于 ResNet18 特征空间）
一次前向传播，同时计算 FID, Precision 和 Recall。

用法：
    from fid_pr_evaluator import compute_fid_and_pr
    fid, precision, recall = compute_fid_and_pr(gen_images, real_images)
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from scipy import linalg
from pathlib import Path

def _load_model(device):
    """内部函数：加载去除了分类头的 ResNet18"""
    default_classifier_path = Path(__file__).parent / 'models' / 'resnet18-f37072fd.pth'
    if not default_classifier_path.exists():
        raise FileNotFoundError(f"找不到 ResNet18 权重文件: {default_classifier_path}")

    model = resnet18(weights=None)
    state_dict = torch.load(default_classifier_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.fc = nn.Identity()   # 去掉分类头，只保留 512 维特征
    return model.to(device).eval()  # 这里必须 return


def _extract_features(images, model, device, batch_size):
    """内部函数：统一的特征提取逻辑，返回 PyTorch Tensor"""
    images = images.to(device)
    # 确保输入为三通道
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    # 调整尺寸到 ResNet18 输入大小 224x224
    images = TF.resize(images, [224, 224], antialias=True)
    # 标准化到 ImageNet 均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std

    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            features.append(model(batch))
            
    return torch.cat(features, dim=0)


def compute_fid_and_pr(gen_images, real_images, device='cuda', batch_size=256, k=3):
    """
    只需提取一次特征，同时计算 FID, Precision 和 Recall。

    返回:
        (fid, precision, recall)
    """
    model = _load_model(device)

    # ==========================================
    # 1. 共享的特征提取阶段 (最耗时的部分只做一次)
    # ==========================================
    phi_real = _extract_features(real_images, model, device, batch_size) # [N, 512]
    phi_gen = _extract_features(gen_images, model, device, batch_size)   # [M, 512]

    # ==========================================
    # 2. 计算 FID (使用 Numpy)
    # ==========================================
    real_feats_np = phi_real.cpu().numpy()
    gen_feats_np = phi_gen.cpu().numpy()

    mu_real, sigma_real = np.mean(real_feats_np, axis=0), np.cov(real_feats_np, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_feats_np, axis=0), np.cov(gen_feats_np, rowvar=False)

    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset) @ (sigma_real + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2.0 * covmean)

    # ==========================================
    # 3. 计算 Precision & Recall (使用 PyTorch 张量距离)
    # ==========================================
    # 寻找流形边界半径 (第 k+1 近的距离)
    dist_real_to_real = torch.cdist(phi_real, phi_real)
    radius_real, _ = torch.kthvalue(dist_real_to_real, k + 1, dim=1) 

    dist_gen_to_gen = torch.cdist(phi_gen, phi_gen)
    radius_gen, _ = torch.kthvalue(dist_gen_to_gen, k + 1, dim=1) 

    # 交叉距离
    dist_gen_to_real = torch.cdist(phi_gen, phi_real) 

    # Precision
    in_real_manifold = (dist_gen_to_real <= radius_real.unsqueeze(0)).any(dim=1)
    precision = in_real_manifold.float().mean().item()

    # Recall
    dist_real_to_gen = dist_gen_to_real.t() 
    in_gen_manifold = (dist_real_to_gen <= radius_gen.unsqueeze(0)).any(dim=1)
    recall = in_gen_manifold.float().mean().item()

    return float(fid), float(precision), float(recall)