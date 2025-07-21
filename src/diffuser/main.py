import torch
from torch import Tensor
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 因为这里的 train=True , 所以读取的是训练集数据
# 如果想读取测试集数据，可以将 train=False
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=torchvision.transforms.ToTensor())
# batch_size表示每个批次的样本数量， shuffle=True表示每个epoch打乱数据顺序
# 这里的 batch_size 可以根据显存大小进行调整
# 如果显存不够，可以将 batch_size 调小
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
x,y = next(iter(train_dataloader))
print(f"Batch shape: {x.shape}, Labels: {y}")
images: Tensor = torchvision.utils.make_grid(x[0:8], nrow=4, padding=0, normalize=True)
print(f"Image shape: {images.shape}")
plt.imshow(images.permute(1,2,0), cmap='gray')
plt.title(f"Label: {y[1]}")
plt.show()

# 定义加噪过程
def add_noise(x: Tensor, t: float): 
    noise = torch.randn_like(x)
    t = t.view(-1, 1, 1, 1)
    return x * (1 - t) + noise * t

figure, axises = plt.subplots(1, 2, figsize=(10, 5))
# 可视化原始图像和加噪图像







