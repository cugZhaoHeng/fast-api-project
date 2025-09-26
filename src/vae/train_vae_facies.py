# -*- coding: utf-8 -*-
"""
3D VAE for Facies Modeling (Mud, Sand, Fluid)
Author: Your Name
Date: 2025-09-26

This script:
- Loads discrete facies models (0.1=mud, 10=sand, 200=fluid)
- Trains a 3D VAE with one-hot encoding and softmax output
- Saves model and generates new realizations
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.logger import create_logger

logger = create_logger(__name__)

# -------------------------------
# 配置参数
# -------------------------------
DATA_DIR = r"../../data/npy_files_F"          # 你的 .npy 文件目录
SAVE_DIR = "./checkpoints_F"                  # 模型保存路径
os.makedirs(SAVE_DIR, exist_ok=True)

INPUT_SHAPE = (16, 64, 64)                  # 模型尺寸
LATENT_DIM = 64                             # 隐空间维度
NUM_CLASSES = 3                             # 泥岩、砂岩、流体
BATCH_SIZE = 4                              # 3D 数据大，batch 要小
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Data dir: {DATA_DIR}")


# -------------------------------
# 1. One-Hot 编码函数
# -------------------------------
def to_onehot(facies_map):
    """
    将原始 facies map (值: 0.1, 10, 200) 转为 one-hot 编码
    输入: (H, W, D)
    输出: (3, H, W, D)
    """
    label_map = np.zeros(facies_map.shape, dtype=np.int64)
    label_map[facies_map == 0.1] = 0   # 泥岩
    label_map[facies_map == 10]  = 1   # 砂岩
    label_map[facies_map == 200] = 2   # 流体

    onehot = np.eye(NUM_CLASSES)[label_map]  # -> (H, W, D, 3)
    onehot = np.transpose(onehot, (3, 0, 1, 2))  # -> (3, H, W, D)
    return onehot


# -------------------------------
# 2. 自定义数据集
# -------------------------------
class FaciesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(path)  # shape: (16, 64, 64)

        # 转为 one-hot
        data_onehot = to_onehot(data) # (3, 16, 64, 64)
        logger.info(f"data_onehot: {data_onehot}")
        return torch.from_numpy(data_onehot).float()


# -------------------------------
# 3. 3D VAE 模型（带 Softmax 输出）
# -------------------------------
class VAE3D_Facies(nn.Module):
    def __init__(self, input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super(VAE3D_Facies, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 编码器：输入 3 通道（one-hot）
        self.encoder = nn.Sequential(
            nn.Conv3d(num_classes, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # 推导特征图大小
        with torch.no_grad():
            dummy = torch.randn(1, num_classes, *input_shape)
            h = self.encoder(dummy)
            self.h_shape = h.shape  # 用于 decode 时 reshape
            h_dim = h.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, num_classes, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)  # 在 channel 维 softmax
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, *self.h_shape[1:])
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


# -------------------------------
# 4. 损失函数（交叉熵 + KL）
# -------------------------------
def vae_loss_facies(recon_probs, x_onehot, mu, log_var, beta=1.0):
    # 转为类别索引
    targets = torch.argmax(x_onehot, dim=1)  # (B, H, W, D)

    # 交叉熵损失
    CE = nn.functional.cross_entropy(recon_probs, targets, reduction='sum')

    # KL 散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return CE + beta * KLD


# -------------------------------
# 5. 生成新模型函数
# -------------------------------
def generate_facies_model(model, device=DEVICE, save_path=None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM).to(device)
        prob_map = model.decode(z)  # (1, 3, 16, 64, 64)

        # 取最大概率类别
        class_map = torch.argmax(prob_map, dim=1).cpu().squeeze().numpy()  # (16, 64, 64)

        # 映射回原始值
        k_map = np.zeros_like(class_map, dtype=np.float32)
        k_map[class_map == 0] = 0.1   # 泥岩
        k_map[class_map == 1] = 10    # 砂岩
        k_map[class_map == 2] = 200   # 流体
        print(k_map)

        if save_path:
            np.save(save_path, k_map)
            print(f"Generated model saved to {save_path}")

        return k_map


# -------------------------------
# 6. 可视化函数
# -------------------------------
def visualize_slice(model, slice_idx=8):
    """可视化中间切片"""
    with torch.no_grad():
        gen = generate_facies_model(model)
        plt.figure(figsize=(8, 6))
        plt.imshow(gen[slice_idx, :, :], cmap='viridis',
                   vmin=0, vmax=250)
        plt.colorbar(label='Value')
        plt.title(f'Generated Facies Model (Z-Slice {slice_idx})\n0.1=Mud, 10=Sand, 200=Fluid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


loss_list = []
# -------------------------------
# 7. 主训练函数
# -------------------------------
def train():
    # 数据集和加载器
    dataset = FaciesDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Loaded {len(dataset)} models.")

    # 模型
    model = VAE3D_Facies().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)  # (B, 3, 16, 64, 64)

            optimizer.zero_grad()
            recon, mu, log_var = model(data)
            loss = vae_loss_facies(recon, data, mu, log_var, beta=0.1)  # beta 可调
            logger.info(f"loss:{loss}")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            logger.info(f"train_loss:{train_loss}")
            loss_list.append(loss.item())

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataset)}] '
                      f'Loss: {loss.item() / len(data):.4f}')
            break
        avg_loss = train_loss / len(dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')


        # 每 20 个 epoch 生成一个模型看看
        if epoch % 20 == 0:
            gen_path = f"./generated_F/generated_model_epoch_{epoch}.npy"
            os.makedirs("generated", exist_ok=True)
            generate_facies_model(model, save_path=gen_path)
            visualize_slice(model)

        # 保存模型
        if epoch % 10 == 0:
            model_path = os.path.join(SAVE_DIR, f"vae_facies_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    print("✅ Training completed!")
    plt.plot(loss_list)
    plt.show()
    plt.close()
    return model


# -------------------------------
# 8. 测试/生成函数
# -------------------------------
def test():
    model = VAE3D_Facies().to(DEVICE)
    model_path = os.path.join(SAVE_DIR, "vae_facies_epoch_100.pth")  # 改成你最好的模型

    if not os.path.exists(model_path):
        print("❌ Model not found. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"✅ Model loaded from {model_path}")

    # 生成 5 个新模型
    os.makedirs("generated", exist_ok=True)
    for i in range(5):
        path = f"generated_F/random_facies_model_{i+1:02d}.npy"
        k_map = generate_facies_model(model, save_path=path)
        print(f"Generated model {i+1}/5: {path}")

    # 可视化一个
    visualize_slice(model)


# -------------------------------
# 9. 主函数
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting VAE Training for Facies Modeling...\n")

    # 选择运行模式
    mode = input("Enter mode ('train' or 'test'): ").strip().lower()

    if mode == "train":
        trained_model = train()
    elif mode == "test":
        test()
    else:
        print("Invalid mode. Use 'train' or 'test'.")