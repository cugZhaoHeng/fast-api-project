import torch
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn

# -----------------------------
# 分类器定义（必须与训练时一致）
# -----------------------------
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

# -----------------------------
# 加载分类器的辅助函数（内部使用）
# -----------------------------
def _load_classifier(path, device):
    """从指定路径加载分类器"""
    if not path.exists():
        raise FileNotFoundError(f"分类器文件 {path} 不存在，请先训练分类器并保存到该路径。")
    try:
        model = MNISTClassifier().to(device)
        state_dict = torch.load(path, map_location=device)
        # 兼容旧保存格式（可能包含 'model_state_dict' 键）
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"加载分类器失败: {e}")

# -----------------------------
# 公开评估函数：只接收图片
# -----------------------------
def evaluate_images(images, device='cuda'):
    """
    评估一批生成图像的质量。

    参数:
        images : torch.Tensor, 形状 (N, 1, 28, 28), 值域 [0,1]
        device : 运行设备

    返回:
        max_conf : 平均置信度
        mean_entropy : 平均熵
        coverage : 类别覆盖率（基于预测类别分布的均匀度）

    异常:
        FileNotFoundError 如果默认分类器文件不存在
        RuntimeError 如果分类器加载失败
    """
    # 默认分类器路径（可根据项目实际调整）
    default_classifier_path = Path(__file__).parent / 'models' / 'mnist_classifier.pth'

    # 加载分类器
    classifier = _load_classifier(default_classifier_path, device)

    # 确保图像形状正确
    images = images.to(device)
    if images.dim() == 3:
        images = images.unsqueeze(1)   # [N,28,28] -> [N,1,28,28]

    with torch.no_grad():
        logits = classifier(images)
        probs = F.softmax(logits, dim=1)

    # 计算指标
    max_conf = probs.max(dim=1)[0].mean().item()
    entropy = - (probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
    pred_dist = probs.mean(dim=0)
    uniform = torch.full_like(pred_dist, 0.1)
    l1_dist = torch.abs(pred_dist - uniform).sum().item()
    coverage = 1.0 - l1_dist / 1.8

    return max_conf, entropy, coverage