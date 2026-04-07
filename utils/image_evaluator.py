import torch
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'

# -----------------------------
# 1. 分类器定义
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
# 2. 【补全】分类器训练逻辑
# -----------------------------
def train_mnist_classifier(data_dir, save_path, epochs=5, batch_size=64, device='cuda'):
    """
    训练一个标准的 MNIST 分类器作为评估的基准。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.MNIST(root=data_dir, train=True, download=False, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = MNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"开始训练评估用分类器...")
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"分类器已保存至: {save_path}")
    return model

# -----------------------------
# 3. 加载分类器的辅助函数
# -----------------------------
def _load_classifier(path, device):
    if not path.exists():
        # 如果文件不存在，自动触发训练（这里假设路径正确）
        print("未发现预训练分类器，准备开始训练...")
        return None 
    try:
        model = MNISTClassifier().to(device)
        state_dict = torch.load(path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"加载分类器失败: {e}")

# -----------------------------
# 4. 公开评估函数
# -----------------------------
def evaluate_images(images, device='cuda', classifier_path=None):
    """
    指标解释：
    1. max_conf (置信度): 越接近 1.0 越好。说明生成的数字非常清晰，分类器很确定它是某个数字。
    2. entropy (熵): 越接近 0 越好。熵高说明图片模糊，分类器觉得它既像3又像8，处于犹豫状态。
    3. coverage (覆盖率): 越接近 1.0 越好。如果VAE只生成数字“1”，覆盖率会很低；如果0-9都生成，覆盖率高。
    """
    if classifier_path is None:
        classifier_path = Path(__file__).parent / 'models' / 'mnist_classifier.pth'
    else:
        classifier_path = Path(classifier_path)

    classifier = _load_classifier(classifier_path, device)
    
    # 如果没找到模型，抛出异常或根据需要处理
    if classifier is None:
        raise FileNotFoundError("请先运行 train_mnist_classifier 训练基准分类器。")

    images = images.to(device)
    if images.dim() == 3:
        images = images.unsqueeze(1)

    with torch.no_grad():
        logits = classifier(images)
        probs = F.softmax(logits, dim=1)

    # A. 计算平均置信度 (Confidence)
    max_conf = probs.max(dim=1)[0].mean().item()

    # B. 计算平均熵 (Entropy)
    entropy = - (probs * (probs + 1e-8).log()).sum(dim=1).mean().item()

    # C. 计算类别覆盖率 (Coverage)
    # 计算这批图片在 10 个类别上的平均分布
    pred_dist = probs.mean(dim=0) # [10]
    # 理想状态是均匀分布 [0.1, 0.1, ..., 0.1]
    uniform = torch.full_like(pred_dist, 0.1)
    # 计算实际分布与理想均匀分布的 L1 距离
    l1_dist = torch.abs(pred_dist - uniform).sum().item()
    # 归一化到 0-1 之间
    coverage = 1.0 - (l1_dist / 1.8) # 1.8 是 L1 距离可能的最大值

    return max_conf, entropy, coverage

# -----------------------------
# 6. 【新增】单张图片预测函数
# -----------------------------
@torch.no_grad()
def predict_single_image(image_input, device='cuda', classifier_path=None):
    """
    输入单张图片，返回预测数字及概率分布。
    
    参数:
        image_input: 可以是图片的 Path 路径，或者形状为 (1, 28, 28) 的 torch.Tensor
        device: 运行设备
        classifier_path: 分类器权重路径
    
    返回:
        predicted_digit (int): 预测的数字
        probabilities (list): 0-9 每个数字的概率 (float)
    """
    # 1. 加载分类器
    if classifier_path is None:
        classifier_path = Path(__file__).parent / 'models' / 'mnist_classifier.pth'
    else:
        classifier_path = Path(classifier_path)

    classifier = _load_classifier(classifier_path, device)
    if classifier is None:
        raise FileNotFoundError("未找到分类器模型，请先训练分类器。")
    classifier.eval()

    # 2. 预处理图片
    if isinstance(image_input, (str, Path)):
        # 如果是路径，加载并转为灰度图，缩放到 28x28
        img = Image.open(image_input).convert('L')
        transform = T.Compose([
            T.Resize((28, 28)),
            T.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0) # 变为 (1, 1, 28, 28)
    elif isinstance(image_input, torch.Tensor):
        img_tensor = image_input
        if img_tensor.dim() == 2: # (28, 28) -> (1, 1, 28, 28)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif img_tensor.dim() == 3: # (1, 28, 28) -> (1, 1, 28, 28)
            img_tensor = img_tensor.unsqueeze(0)
    else:
        raise ValueError("不支持的输入类型，请输入路径或 Tensor。")

    # 3. 前向传播
    img_tensor = img_tensor.to(device)
    logits = classifier(img_tensor)
    probs = F.softmax(logits, dim=1) # 得到概率分布

    # 4. 获取结果
    probs_list = probs.squeeze().cpu().tolist() # 转为 Python 列表
    predicted_digit = torch.argmax(probs, dim=1).item()

    return predicted_digit, probs_list

# -----------------------------
# 7. 打印预测结果的辅助工具
# -----------------------------
def print_prediction_report(digit, probs):
    """漂亮地打印预测结果"""
    print("\n" + "="*30)
    print(f"  预测结果: 【 {digit} 】")
    print("-"*30)
    print("  数字 |   概率")
    for i, p in enumerate(probs):
        # 使用进度条直观展示概率
        bar = "#" * int(p * 20)
        print(f"   {i}   |  {p:6.2%}  {bar}")
    print("="*30 + "\n")

# -----------------------------
# 测试代码示例
# -----------------------------
if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 假设你有一张 VAE 生成的图片或本地图片
    # 这里我们随机生成一个张量作为演示
    test_img = torch.rand(1, 28, 28) 
    
    try:
        digit, probs = predict_single_image(test_img, device=DEVICE)
        print_prediction_report(digit, probs)
    except Exception as e:
        print(f"预测失败: {e}")
    
    # 获取一个 batch
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform),
        batch_size=128, shuffle=True
    )
    images, labels = next(iter(train_loader))
    # images shape: [128, 1, 28, 28] (batch_size, channel, height, width)

    # 取出第一张图片
    single_image = images[0]  # shape: [1, 28, 28]
    single_label = labels[0]

    print(f"图片张量形状: {single_image.shape}")
    print(f"标签: {single_label}")
    
    try:
        digit, probs = predict_single_image(single_image, device=DEVICE)
        print_prediction_report(digit, probs)
    except Exception as e:
        print(f"预测失败: {e}")

