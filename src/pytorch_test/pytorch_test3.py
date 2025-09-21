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

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1: Linear = Linear(in_features=28*28, out_features=200, bias=True)
        self.activation: ReLU = ReLU()
        self.fc2: Linear = Linear(in_features=200, out_features=100, bias=True)
        self.fc3 : Linear = Linear(in_features=100, out_features=1, bias=True)
        self.loss_function = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

def train():
    net = NeuralNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_dataset: Dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                                        download=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    loss_list = []

    for i in range(5):
        for images, labels in tqdm(train_dataloader):
            train_images: Tensor = images.to(device=device)
            train_labels: Tensor = labels.to(device=device)

            print(f"train_labels: {train_labels}")
            predictions: Tensor = net.forward(train_images)
            predictions = predictions.squeeze()
            print(f'predictions: {predictions}')

            loss = net.loss_function(predictions, train_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"train_loss: {loss.item():.6f}")
            loss_list.append(loss.item())

    torch.save(net.state_dict(), f'model_3.pth')

    # plt.plot(loss_list)
    # plt.show()

def test():
    with open(file="model_3.pth", mode='rb') as f:
        module = torch.load(f=f, map_location=device)
        test_dataset: Dataset = torchvision.datasets.MNIST(root="../../data", train=False, download=False, transform=transforms.ToTensor())
        test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

        model = NeuralNetwork()
        model.load_state_dict(state_dict=module)
        model.eval()
        all_predication =[]
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                predict: Tensor = model.forward(images)
                predict = predict.squeeze()
                predict = torch.round(predict).int()
                print(f"labels: {labels}")
                print(f"predict: {predict}")
                all_predication.append(predict)
                all_labels.append(labels)
            all_predication = torch.cat(all_predication)
            all_labels = torch.cat(all_labels)
            print(f"all_predication: {all_predication}")
            print(f"all_labels: {all_labels}")
            accuracy = (all_predication == all_labels).float().mean().item()
            print(f"测试准确率: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    # train()
    test()
