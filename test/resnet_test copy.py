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

# ==========================================
# 2. 搭建用于 MNIST 的 Mini-ResNet
# ==========================================
class ResNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetMNIST, self).__init__()
        
        # 初始层：把 MNIST 的 1 个灰度通道变成 16 个通道
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 开始堆叠残差块！
        # Layer 1: 图像大小不变 (28x28)，通道数 16->16
        self.layer1 = self._make_layer(16, num_blocks=2, stride=1)
        
        # Layer 2: 图像下采样 (14x14)，通道数 16->32
        self.layer2 = self._make_layer(32, num_blocks=2, stride=2)
        
        # 全局平均池化 (把 14x14 压缩成 1x1 的特征点)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层输出分类结果 (32 -> 10类)
        self.fc = nn.Linear(32, num_classes)

    # 这是一个辅助函数，用来连续生成多个残差块
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # 例如 [2, 1]
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 过残差层
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平
        x = self.fc(x)
        
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

if __name__ == '__main__':
    main()