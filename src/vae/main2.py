import os
# 在导入 torch 或 numpy 等库之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# 1. 定义 VAE 模型
# -----------------------------------
class VAE(nn.Module):
    '''
    关于这个init代码，有两个地方需要搞清楚：
    1.VAE的流程，也就是输入数据在里面的流转情况，这个直接关系到pytorch中模型的定义；
    2.Sequential的实现原理，以及在nn中它和nn.Linear以及其他神经网络构造函数的区别
    3.我在创建一个神经网络的时候，怎么才能知道自己要按照什么步骤创建，需要引入哪些网络？
    '''
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 输入的维度大小，默认是784维
        self.input_dim = input_dim
        # 潜在空间的维度大小，默认是400维
        self.latent_dim = latent_dim

        # 编码器：将图像映射到均值和方差，Sequential和自定义的神经网络有什么区别？
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Linear的含义是什么？
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值 μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log(σ²)

        # 解码器：从隐变量重建图像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出像素值在 [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)          # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)             # ε ~ N(0,1)
        return mu + std * eps                   # z = μ + σ * ε

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# -----------------------------------
# 2. 损失函数：重建误差 + KL 散度
# -----------------------------------
def vae_loss(recon_x, x, mu, log_var):
    # 重建误差：二值交叉熵（BCE），衡量图像差异
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度：让隐变量分布接近标准正态分布
    # 公式: 0.5 * sum(log_var.exp() + mu^2 - 1 - log_var)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

# -----------------------------------
# 3. 数据加载
# -----------------------------------
transform = transforms.ToTensor()  # 将图像转为 [0,1] 的张量
train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -----------------------------------
# 4. 初始化模型、优化器
# -----------------------------------
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------------
# 5. 训练循环
# -----------------------------------
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(-1, 784)  # 展平成 784 维向量

        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.4f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# -----------------------------------
# 6. 生成并可视化图片
# -----------------------------------
def generate_images():
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, model.latent_dim).to(device)  # 从 N(0,1) 采样 16 个隐变量
        sample = model.decode(z).cpu()                    # 解码生成图片
        sample = sample.view(16, 1, 28, 28)               # 恢复为 28x28 图像

        # 可视化
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axes[i // 4, i % 4]
            ax.imshow(sample[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# -----------------------------------
# 7. 开始训练
# -----------------------------------
if __name__ == "__main__":
    for epoch in range(1, 11):
        train(epoch)
        if epoch % 5 == 0:
            generate_images()