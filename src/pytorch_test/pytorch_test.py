import os
import sys

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import Linear, ReLU, Sigmoid, Tanh
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def test1():
    # 测试 Linear 的用法
    # 随机创建一个数组，大小是1*10
    x: Tensor = torch.randn(size=(1, 10))
    print(x)
    print(f"x的形状：{x.shape}")

    fc1: Linear = Linear(in_features=10, out_features=3, bias=True)
    print(type(fc1))
    y: Tensor = fc1(x)
    print(f"y的输出：{y}")
    activate_function = ReLU()
    # 经过激活函数
    y = activate_function(y)
    print(f"y shape: {y.shape}")
    print(f"y content: {y}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1: Linear = Linear(in_features=28*28, out_features=200, bias=True)
        self.activation: ReLU = ReLU()
        self.fc2: Linear = Linear(in_features=200, out_features=10, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def train():
    net = NeuralNetwork()
    net.to(torch.device('cuda'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_dataset: Dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                                        download=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    loss_list = []

    for images, labels in train_dataloader:
        train_images: Tensor = images.to(device=device)
        train_labels: Tensor = labels.to(device=device)
        print(f'train_images.shape: {train_images.shape}')

        print(f"train_labels: {train_labels}")
        print(f"train_labels.shape: {train_labels.shape}")

        predictions: Tensor = net.forward(train_images)
        print(f'predictions: {predictions}')
        print(f'predictions.shape: {predictions.shape}')

        pre = predictions.softmax(dim=1)
        print(f'pre: {pre}')

        loss = net.loss_function(predictions, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"train_loss: {loss.item():.6f}")
        loss_list.append(loss.item())

    # torch.save(net.state_dict(), f'model.pth')

    plt.plot(loss_list)
    plt.show()

def test():
    with open(file="model.pth", mode='rb') as f:
        module = torch.load(f=f)
        print(module)

if __name__ == '__main__':
    test()
