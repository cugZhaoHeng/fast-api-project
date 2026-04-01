import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# 路径配置
# ----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
MNIST_DIR = PROJECT_ROOT_DIR / 'data' / 'MNIST'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = MODEL_DIR / "mnist_classifier.pth"   # 分类器保存路径
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"      # VAE 模型路径（无关，保留）

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
device = init_gpu_environment()

batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# ----------------------------
# 分类器定义
# ----------------------------
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


def train_mnist_classifier(train_loader, test_loader, epochs=6, lr=1e-3, save_path=None):
    """
    训练分类器，并可选择保存模型。
    返回 (训练好的模型, 测试集准确率)
    """
    clf = MNISTClassifier().to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        clf.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = ce(clf(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        logger.info(f"Epoch {epoch+1}/{epochs} loss: {total_loss/len(train_loader.dataset):.4f}")

    # 测试
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = clf(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    logger.info(f"Classifier test accuracy: {acc:.4f}")

    # 保存模型
    if save_path is not None:
        torch.save({
            'model_state_dict': clf.state_dict(),
            'test_accuracy': acc,
        }, save_path)
        logger.info(f"Classifier saved to {save_path}")

    return clf, acc


def load_classifier(path, device='cpu'):
    """
    从 .pth 文件加载分类器。
    返回模型对象，并设置为 eval 模式。
    """
    clf = MNISTClassifier().to(device)
    checkpoint = torch.load(path, map_location=device)
    clf.load_state_dict(checkpoint['model_state_dict'])
    clf.eval()
    logger.info(f"Classifier loaded from {path}, test accuracy: {checkpoint.get('test_accuracy', 'unknown')}")
    return clf


# ----------------------------
# 评估 VAE 生成图像（需用到分类器）
# ----------------------------
@torch.no_grad()
def evaluate_generated_images(model_eval, latent_dim, n_gen=10000, batch_size=256,
                              mode="normal", ref_loader=None, classifier=None, classifier_path=None):
    """
    评估 VAE 生成的图像质量（置信度、熵、类别覆盖率）。

    参数：
        model_eval   : VAE 的 EMA 模型（已 eval）
        latent_dim   : 隐空间维度
        n_gen        : 生成图像总数
        batch_size   : 生成时的批次大小
        mode         : "normal" 或 "agg"（聚合后验）
        ref_loader   : 当 mode="agg" 时，用于采样参考数据的 DataLoader
        classifier   : 可选的已训练的分类器（若提供则直接使用）
        classifier_path : 若 classifier 为 None，则从该路径加载分类器
                          默认为 CLASSIFIER_PATH

    返回：
        max_conf, mean_entropy, coverage
    """
    # 加载分类器
    if classifier is None:
        path = classifier_path if classifier_path is not None else CLASSIFIER_PATH
        if not path.exists():
            raise FileNotFoundError(f"Classifier not found at {path}. Please train and save it first.")
        clf = load_classifier(path, device=device)
    else:
        clf = classifier
        clf.eval()

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
        # 确保形状符合分类器要求
        if gen.dim() == 2:
            gen = gen.view(gen.size(0), 1, 28, 28)
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

def evaluate_images(images, classifier=None, classifier_path=None, device='cuda'):
    """
    评估一批生成图像的质量。

    参数:
        images : torch.Tensor, 形状 (N, 1, 28, 28), 值域应在 [0,1] 之间（建议用 sigmoid 输出）。
        classifier : 可选，已加载的分类器模型。若为 None，则自动从 classifier_path 加载。
        classifier_path : 分类器权重路径。若 classifier 为 None 且路径存在，则加载；否则需要先训练。
        device : 运行设备。

    返回:
        max_conf : 平均置信度（最大 softmax 概率）
        mean_entropy : 平均熵
        coverage : 类别覆盖率（基于预测类别分布的均匀度）
    """
    # 1. 获取分类器
    if classifier is None:
        if classifier_path is None:
            # 若未提供路径，使用默认路径（可根据实际情况调整）
            classifier_path = Path(__file__).parent / 'models' / 'mnist_classifier.pth'
        if not classifier_path.exists():
            # 若分类器不存在，尝试训练一个（需要训练数据，这里假设外部已训练）
            raise FileNotFoundError(f"分类器文件 {classifier_path} 不存在，请先训练分类器。")
        classifier = load_classifier(classifier_path, device=device)
    classifier.eval()

    # 2. 将图像移动到设备
    images = images.to(device)
    # 确保图像形状正确（可能输入缺少通道维度）
    if images.dim() == 3:
        images = images.unsqueeze(1)  # [N,28,28] -> [N,1,28,28]

    # 3. 分类器前向传播
    with torch.no_grad():
        logits = classifier(images)          # [N,10]
        probs = F.softmax(logits, dim=1)     # [N,10]

    # 4. 计算指标
    max_conf = probs.max(dim=1)[0].mean().item()          # 平均置信度
    entropy = - (probs * (probs + 1e-8).log()).sum(dim=1).mean().item()  # 平均熵
    # 覆盖率：基于预测类别分布与均匀分布的 L1 距离
    pred_dist = probs.mean(dim=0)            # 预测类别概率分布
    uniform = torch.full_like(pred_dist, 0.1)
    l1_dist = torch.abs(pred_dist - uniform).sum().item()
    coverage = 1.0 - l1_dist / 1.8           # 最大 L1 距离为 1.8（当所有样本都集中在某一类时）

    return max_conf, entropy, coverage
# ----------------------------
# 主程序：训练/加载分类器并保存
# ----------------------------
if __name__ == "__main__":
    # 检查分类器是否已存在
    if CLASSIFIER_PATH.exists():
        logger.info(f"Loading existing classifier from {CLASSIFIER_PATH}")
        clf = load_classifier(CLASSIFIER_PATH, device=device)
        # 可选：再次测试一下加载的模型，确认正确
        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = clf(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        logger.info(f"Loaded classifier test accuracy: {correct/total:.4f}")
    else:
        logger.info("No classifier found. Training a new one...")
        clf, acc = train_mnist_classifier(train_loader, test_loader, epochs=10, lr=1e-3,
                                          save_path=CLASSIFIER_PATH)
        logger.info(f"Trained classifier test accuracy: {acc:.4f}")