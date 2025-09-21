import os
import sys

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import Linear, ReLU, Sigmoid, Tanh
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from utils import logger

logger = logger.create_logger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_tensor_2_image(x: Tensor, filepath: str):
    plt.figure(figsize=(5, 5))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(x[0][i].cpu().detach().numpy(), cmap='gray')
    plt.savefig(fname=filepath, bbox_inches="tight", pad_inches=0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=10)

        self.activate_function = ReLU()
        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        # convert_tensor_2_image(x=x, filepath=r"images/conv1.png")

        x = self.activate_function(x)
        x = self.pool1(x)

        x = self.conv2(x)
        # logger.info(f"x conv2: {x.shape}")
        # convert_tensor_2_image(x=x, filepath=r"images/conv2.png")
        x = self.activate_function(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


def train():
    net = NeuralNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_dataset: Dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                                        download=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    loss_list = []

    for i in range(1):
        for images, labels in tqdm(train_dataloader):
            train_images: Tensor = images.to(device=device)
            train_labels: Tensor = labels.to(device=device)

            predictions: Tensor = net.forward(train_images)

            loss = net.loss_function(predictions, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"train_loss: {loss.item():.6f}")
            loss_list.append(loss.item())

    torch.save(net.state_dict(), f'model_4.pth')

    plt.plot(loss_list)
    plt.savefig("images/loss.png")


def test():
    with open(file="model_4.pth", mode='rb') as f:
        module = torch.load(f=f, map_location=device)
        test_dataset: Dataset = torchvision.datasets.MNIST(root="../../data", train=False, download=False,
                                                           transform=transforms.ToTensor())
        test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

        model = NeuralNetwork()
        model.load_state_dict(state_dict=module)
        model.eval()
        all_predication = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                predict: Tensor = model.forward(images)
                print(f"predict: {predict}")
                predict = torch.argmax(predict, dim=1) # dim=1 每行的最大值索引, dim=0 每列的最大值索引
                print(f"labels: {labels}")
                print(f"predict: {predict}")
                all_predication.append(predict)
                all_labels.append(labels)
                break
            all_predication = torch.cat(all_predication)
            all_labels = torch.cat(all_labels)
            print(f"all_predication: {all_predication}")
            print(f"all_labels: {all_labels}")
            accuracy = (all_predication == all_labels).float().mean().item()
            print(f"测试准确率: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    # train()
    test()
