import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import Linear, ReLU
from torch import Tensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = Linear(in_features=28 * 28, out_features=200, bias=True)
        self.activation = ReLU()
        self.fc2 = Linear(in_features=200, out_features=10, bias=True)
        # 使用交叉熵损失而不是MSE，更适合分类问题
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        # 将图像展平为向量 (batch_size, 28*28)
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = NeuralNetwork()
    net.to(device)
    # 如果你使用的是PyTorch 2.0+，可以取消下面的注释
    # net = torch.compile(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_dataset = torchvision.datasets.MNIST(
        root=r'D:\gitee\fast-api-project\data',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  # 增大batch size

    # 训练多个epoch
    for epoch in range(5):
        total_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            train_images = images.to(device=device)
            train_labels = labels.to(device=device)

            # 前向传播
            predictions = net(train_images)

            # 计算损失 - 使用交叉熵损失，不需要one-hot编码
            loss = net.loss_function(predictions, train_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.6f}")

        print(f"Epoch {epoch} completed. Average loss: {total_loss / len(train_dataloader):.6f}")

    # 测试模型
    test_dataset = torchvision.datasets.MNIST(
        root=r'D:\gitee\fast-api-project\data',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')