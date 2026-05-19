from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
IMAGE_DIR = CURRENT_DIR / 'images'
DATA_DIR = PROJECT_ROOT_DIR / 'data' 

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
from utils.mnist_util import load_mnist_images, load_mnist_labels
from utils.date_util import get_current_time

# ==========================================
# 1. 定义残差块 (Residual Block)
# ==========================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # --- 实习生的工作：F(x) ---
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # --- 老板的复印件：x (Shortcut/捷径) ---
        self.shortcut = nn.Sequential()
        # 【关键细节】：如果 stride != 1 (图像变小了)，或者通道数变了
        # 老板手里的原件 x 就和实习生交上来的 F(x) 尺寸对不上了，没法相加！
        # 解决办法：用一个 1x1 的卷积帮 x 调整一下通道数和尺寸，这叫 Projection Shortcut。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1. 老板先把原稿存起来（走捷径）
        identity = self.shortcut(x) 
        
        # 2. 实习生开始处理（走网络层提取特征，学习残差 F(x)）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 3. 【见证奇迹的时刻】：F(x) + x
        out += identity  # 将实习生提取的差异，叠加到老板的原稿上
        
        # 4. 最后过一次激活函数
        out = self.relu(out)
        
        return out


# (假设前面的 BasicBlock 已经定义好了，保持不变)
# class BasicBlock(nn.Module): ...

# ==========================================
# 新增：面向对象封装的 ResNet Stage (阶段)
# ==========================================
class ResNetStage(nn.Module):
    """
    一个 Stage 包含多个 BasicBlock。
    逻辑非常清晰：第一个 Block 负责调整通道数和尺寸，剩下的 Block 尺寸/通道不变。
    """
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(ResNetStage, self).__init__()
        
        # 1. 明确定义第一个 Block (先锋)
        # 它负责把 in_channels 变成 out_channels，并应用传入的 stride
        self.first_block = BasicBlock(in_channels, out_channels, stride=stride)
        
        # 2. 明确定义剩下的 Blocks (跟随者)
        # 因为第一块已经把通道变成了 out_channels，所以剩下的块：
        # 输入和输出都是 out_channels，且 stride 永远是 1 (不改变图片大小)
        rest_blocks = []
        for _ in range(num_blocks - 1):
            rest_blocks.append(
                BasicBlock(in_channels=out_channels, out_channels=out_channels, stride=1)
            )
        
        # 将剩下的块打包进 Sequential
        self.rest_blocks = nn.Sequential(*rest_blocks)

    def forward(self, x):
        x = self.first_block(x)      # 先过第一个块，完成可能的下采样和通道扩张
        x = self.rest_blocks(x)      # 再过剩下的块，深化特征
        return x

# ==========================================
# 重构后的 ResNet 主网络：极其干净！
# ==========================================
class ResNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetMNIST, self).__init__()
        
        # 1. 初始层 (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 2. 核心：堆叠 Stages (一目了然的面向对象调用)
        # Stage 1: 接收16通道，输出16通道，总共2个块，步长1(大小不变)
        self.stage1 = ResNetStage(in_channels=16, out_channels=16, num_blocks=2, stride=1)
        
        # Stage 2: 接收16通道，输出32通道，总共2个块，步长2(大小缩小一半)
        self.stage2 = ResNetStage(in_channels=16, out_channels=32, num_blocks=2, stride=2)
        
        # 3. 分类头 (Head)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)         # 初始处理
        x = self.stage1(x)       # 第一阶段
        x = self.stage2(x)       # 第二阶段
        x = self.classifier(x)   # 分类输出
        return x

# ==========================================
# 3. 数据加载与训练流程
# ==========================================
def main():
    # 检测是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理 (MNIST 标准均值和方差)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ResNetMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 简易训练循环 (跑 2 个 Epoch 意思一下)
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零 -> 前向传播 -> 计算损失 -> 反向传播 -> 更新权重
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 200 == 199:    # 每 200 个 batch 打印一次
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/200:.4f}")
                running_loss = 0.0

    # 测试环节
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 测试时不计算梯度
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"\nAccuracy of the Custom ResNet on the 10000 test images: {100 * correct / total:.2f}%")
    
    # 1. 实例化你的模型
    model = ResNetMNIST()

    # 2. 伪造一个输入数据 (Batch Size=1, Channel=1, Width=28, Height=28)
    dummy_input = torch.randn(1, 1, 28, 28)

    # 3. 将模型导出为 ONNX 格式
    torch.onnx.export(model, dummy_input, CURRENT_DIR / "resnet_mnist.onnx", 
                    export_params=True, 
                    opset_version=17, 
                    do_constant_folding=True)
    print("ONNX 文件导出成功！")
    
    from torchinfo import summary

    model = ResNetMNIST()
    # 告诉它输入数据的形状是 (Batch, Channel, H, W)
    summary(model, input_size=(1, 1, 28, 28))

if __name__ == '__main__':
    main()