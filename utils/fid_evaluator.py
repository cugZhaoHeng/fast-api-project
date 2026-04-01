"""
FID 评估模块（使用 ResNet18 特征提取）
用法：
    from fid_evaluator import compute_fid
    fid = compute_fid(gen_images, real_images, resnet_weights_path='models/resnet18-f37072fd.pth')
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
from scipy import linalg
from pathlib import Path
from tqdm import tqdm

def compute_fid(gen_images, real_images, device='cuda', resnet_weights_path=None, batch_size=256, verbose=True):
    """
    计算生成图像与真实图像之间的 FID 分数。

    参数:
        gen_images : torch.Tensor, 形状 (N, C, H, W), 值域 [0,1]
        real_images : torch.Tensor, 形状 (M, C, H, W), 值域 [0,1]
        device : 计算设备
        resnet_weights_path : 可选，本地 ResNet18 权重文件路径。若为 None，则使用 torchvision 预训练权重。
        batch_size : 特征提取时的批次大小（用于避免内存溢出）
        verbose : 是否打印进度信息

    返回:
        fid : float
    """
    # 1. 构建特征提取器（ResNet18，移除分类头）
    if resnet_weights_path is not None and Path(resnet_weights_path).exists():
        # 加载本地权重
        model = resnet18(weights=None)
        state_dict = torch.load(resnet_weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        if verbose:
            print(f"使用本地 ResNet18 权重: {resnet_weights_path}")
    else:
        # 使用 torchvision 预训练权重
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if verbose:
            print("使用 torchvision 预训练 ResNet18 权重")
    model.fc = nn.Identity()   # 去掉分类头，只保留特征
    model = model.to(device).eval()

    # 2. 特征提取函数（支持批量）
    def extract_features(images, desc):
        images = images.to(device)
        # 确保输入为三通道
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        # 调整尺寸到 ResNet18 输入大小 224x224
        images = TF.resize(images, [224, 224], antialias=True)
        # 标准化到 ImageNet 均值和标准差
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        images = (images - mean) / std

        features = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                feat = model(batch)
                features.append(feat.cpu().numpy())
                if verbose:
                    print(f"\r{desc}: {min(i+batch_size, len(images))}/{len(images)}", end='')
        if verbose:
            print()
        return np.concatenate(features, axis=0)

    # 3. 提取真实图像特征
    if verbose:
        print("提取真实图像特征...")
    real_feats = extract_features(real_images, "真实图像")

    # 4. 提取生成图像特征
    if verbose:
        print("提取生成图像特征...")
    gen_feats = extract_features(gen_images, "生成图像")

    # 5. 计算均值和协方差
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)

    # 6. 计算 FID
    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset) @ (sigma_real + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2.0 * covmean)

    return float(fid)