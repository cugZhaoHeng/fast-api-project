import os
from pathlib import Path
import torch
from torch import Tensor
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.logger import create_logger

logger = create_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARENT_DIR = Path(__file__).parent
MNIST_PARENT_DIR = PROJECT_ROOT / "data"
model_dir = PARENT_DIR / "models"
image_dir = PARENT_DIR / "images"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
model_path = model_dir / "diffusion_model.pth"
loss_image_path = image_dir / "loss_image.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 因为这里的 train=True , 所以读取的是训练集数据
# 如果想读取测试集数据，可以将 train=False
train_dataset = torchvision.datasets.MNIST(root=MNIST_PARENT_DIR, train=True, download=False,
                                           transform=torchvision.transforms.ToTensor())
# batch_size表示每个批次的样本数量， shuffle=True表示每个epoch打乱数据顺序
# 这里的 batch_size 可以根据显存大小进行调整
# 如果显存不够，可以将 batch_size 调小
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        ])

        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=5, stride=1, padding=2),
        ])
        self.activation_function = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for idx, layer in enumerate(self.down_layers):
            x = layer(x)
            x = self.activation_function(x)
            if idx < 2:
                h.append(x)
                x = self.downscale(x)

        for idx, layer in enumerate(self.up_layers):
            if idx > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.activation_function(layer(x))
        return x

# 定义加噪过程
def add_noise(x: Tensor, t):
    noise = torch.randn_like(x)
    t = t.view(-1, 1, 1, 1)
    return x * (1 - t) + noise * t

# figure, axises = plt.subplots(1, 2, figsize=(10, 5))
# 可视化原始图像和加噪图像

def train(model, device, train_dataloader, optimizer, epochs):
    loss_list = []
    global start_epoch
    if model_path.exists():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss_list = checkpoint["loss"]
        logger.info(f"模型已存在， epoch:{start_epoch}")


    for epoch in range(start_epoch, start_epoch + epochs):
        logger.info(f"[epoch: {epoch}/{start_epoch + epochs}]")
        avg_loss = 0
        for idx, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device)
            noise_x = add_noise(x, noise_amount)
            prediction = model.forward(noise_x)
            loss = F.mse_loss(prediction, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = (idx * avg_loss + loss.item()) / (idx + 1)
        loss_list.append(avg_loss)
    plt.plot(loss_list)
    plt.savefig(loss_image_path)
    torch.save({
        'epoch': start_epoch + epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_list
    }, model_path)
    logger.info(f"训练结束")

def test():
    model = BasicUNet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_model = input("请选择生成还是对比：(1) generate (2) compare: ")
    if test_model == "generate":
        x = torch.randn(1,1,28,28)
        with torch.no_grad():
            y = model.forward(x)
            y = y.detach().cpu().numpy()
            plt.imshow(y.squeeze(), cmap="gray")
            plt.show()
    elif test_model == "compare":
        x,y = next(iter(train_dataloader))
        x = x[: 8]
        amount = torch.linspace(0, 1, x.shape[0])
        noise_x = add_noise(x, amount)
        with torch.no_grad():
            y = model.forward(noise_x)
            # y = y.detach().cpu().numpy()
            fig, axis = plt.subplots(3, 1, figsize=(12, 7))
            axis[0].imshow(torchvision.utils.make_grid(x)[0].clip(0,1), cmap="gray")
            axis[1].imshow(torchvision.utils.make_grid(noise_x)[0].clip(0,1), cmap="gray")
            axis[2].imshow(torchvision.utils.make_grid(y)[0].clip(0,1), cmap="gray")
            plt.show()


if __name__ == '__main__':
    # model = BasicUNet()
    # x = torch.randn(1, 1, 28, 28)
    # y = model(x)
    # y = y.detach().numpy()
    # logger.info(f"Output shape: {y.shape}")
    # logger.info(f"Output: {y}")
    # plt.imshow(y[0][0])
    # plt.show()

    mode = input("请输入要执行的函数：(1) train (2) test: ")
    if mode == "train":
        model = BasicUNet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train(model=model, device = device, train_dataloader=train_dataloader, optimizer=optimizer, epochs=3)
    elif mode == "test":
        test()








