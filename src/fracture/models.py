import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CrackDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')  # 如果图像是灰度图像，请使用 'L' 而不是 'RGB'
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 根据实际情况调整mean和std
])

img_dir = 'D:\\temp\\diffusion\\images'  # 确保路径正确无误
dataset = CrackDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

batch_size = 16
shuffle = True
num_workers = 4  # 多进程加载数据（根据CPU核心数调整）

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)

class VAE(nn.Module):
    def __init__(self, img_size=256, z_dim=128):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 32 -> 16
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 16 * 16, z_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, z_dim)

        # 解码器
        self.decoder_input = nn.Linear(z_dim, 256 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (256, 16, 16))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 128 -> 256
            nn.Sigmoid()  # 输出 [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = self.unflatten(h)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# 初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(z_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(100):
    for data in dataloader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()/data.size(0):.4f}")

# 保存模型
torch.save(model.state_dict(), "crack_vae.pth")