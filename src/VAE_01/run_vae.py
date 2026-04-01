import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
MNIST_DIR = PROJECT_ROOT_DIR / 'data' / 'MNIST'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment

# --- 1. 参数设置 ---
device = init_gpu_environment()
batch_size = 128
latent_channels = 4
latent_spatial_size = 4
NUM_EPOCHS = 1
lr = 1e-3
KL_WEIGHT = 1.5

# --- 2. 加载 MNIST 数据集 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 3. 定义卷积 VAE 模型 ---
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # 编码器：(1, 28, 28) -> (latent_channels, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 4, 4)
            nn.ReLU(),
        )

        # 用于计算 mu 和 logvar 的卷积层（保持空间结构）
        # 我们用卷积层代替全连接层，输出依然是 (C, 4, 4)
        self.fc_mu = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(64, latent_channels, kernel_size=1)

        # 解码器：从 (latent_channels, 4, 4) 还原到 (1, 28, 28)
        self.decoder_input = nn.Conv2d(latent_channels, 64, kernel_size=1)

        self.decoder = nn.Sequential(
            # output_padding 用于精确匹配尺寸：4->7->14->28
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # -> (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- 4. 损失函数 ---
def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    # 重建损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL 散度 (注意：这里建议权重保持 1.0 以保证生成效果)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 全部除以 batch_size 得到样本均值
    return (BCE + KLD) / batch_size, BCE / batch_size, KLD / batch_size

# --- 5. 训练模型 ---
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()


def train_model():
    loss_list, bce_list, kld_list = [], [], []
    start_epoch, train_times = 0, 0
    
    if LATEST_MODEL_PATH.exists():
        # 增加 weights_only=False 解决新版 PyTorch 加载报错
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        bce_list = checkpoint.get("bce_losses", [])
        kld_list = checkpoint.get("kld_losses", [])
        start_epoch = checkpoint["epoch"]
        train_times = checkpoint.get("train_times", 0)
        logger.info(f"已加载模型, 起始 epoch={start_epoch}")
    else:
        logger.info("第一次训练模型")

    model.train()
    end_epoch = start_epoch + NUM_EPOCHS
    
    for epoch in range(start_epoch, end_epoch):
        avg_loss, avg_bce, avg_kld = 0, 0, 0
        temp_mu, temp_std, temp_logvar = [], [], []

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # 获取分解后的损失
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            avg_bce = (avg_bce * batch_idx + bce.item()) / (batch_idx + 1)
            avg_kld = (avg_kld * batch_idx + kld.item()) / (batch_idx + 1)
            
            temp_mu.append(mu.mean().item())
            temp_std.append(mu.std().item())
            temp_logvar.append(logvar.mean().item())

        loss_list.append(avg_loss)
        bce_list.append(avg_bce)
        kld_list.append(avg_kld)
        
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f})")

    # 保存模型，强制转换 numpy 类型为原生 float
    torch.save({
        'epoch': end_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': loss_list,
        'bce_losses': bce_list,
        'kld_losses': kld_list,
        'train_times': train_times + 1,
        'mu_mean': float(np.mean(temp_mu)),
        'mu_std': float(np.mean(temp_std)),
        'logvar_mean': float(np.mean(temp_logvar))
    }, f=LATEST_MODEL_PATH)


    plt.figure(figsize=(15, 5))
    titles = ['Total Loss', 'BCE (Reconstruction)', 'KLD (Latent Regularization)']
    data_to_plot = [loss_list, bce_list, kld_list]
    colors = ['b', 'g', 'r']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        curr_data = data_to_plot[i]
        epochs_range = np.arange(1, len(curr_data) + 1)
        plt.plot(epochs_range, curr_data, color=colors[i], alpha=0.3)
        
        # 20点采样逻辑
        num_pts = min(len(curr_data), 20)
        indices = np.linspace(0, len(curr_data)-1, num_pts, dtype=int)
        plt.scatter(epochs_range[indices], np.array(curr_data)[indices], color=colors[i], s=30)
        
        plt.title(titles[i])
        plt.xlabel("epochs")
        plt.grid(True, linestyle='--', alpha=0.5)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"loss_metrics_{timestamp}_epoch_{end_epoch}.png")
    plt.close()
    logger.info("指标图表已保存。")

def show_model_status():
    """查看当前模型参数及隐空间分布状态"""
    if not LATEST_MODEL_PATH.exists():
        print("\n[提示] 尚未发现模型文件。")
        return

    try:
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
        
        epoch = checkpoint.get('epoch', 0)
        train_times = checkpoint.get('train_times', 0)
        loss_list = checkpoint.get('train_losses', [])
        latest_loss = loss_list[-1] if loss_list else "N/A"
        
        # 提取隐空间统计量
        mu_mean = checkpoint.get('mu_mean', "N/A")
        mu_std = checkpoint.get('mu_std', "N/A")
        logvar_mean = checkpoint.get('logvar_mean', "N/A")

        print("\n" + "="*40)
        print(f"{'模型状态报告':^36}")
        print("-" * 40)
        print(f" 已训练总轮数:    {epoch}")
        print(f" 累计训练次数:    {train_times}")
        if isinstance(latest_loss, float):
            print(f" 最近平均 Loss:   {latest_loss:.4f}")
        
        print("-" * 40)
        print(f"{'隐空间分布 (Latent Space Check)':^36}")
        # 核心调试指标
        if isinstance(mu_mean, (float, int)):
            print(f" Mu 均值 (应接近 0):   {mu_mean:+.4f}")
            print(f" Mu 标准差 (应接近 1): {mu_std:.4f}")
            print(f" LogVar 均值 (应负数): {logvar_mean:.4f}")
            
            # 简单的自动诊断
            if abs(mu_mean) > 0.5 or abs(mu_std - 1.0) > 0.5:
                print("\n[诊断结论]: 隐空间未对齐标准正态分布。")
                print(" -> 原因: KLD 权重可能太低 (当前 0.1)。")
                print(" -> 结果: Generate 模式无法生成有效数字。")
            else:
                print("\n[诊断结论]: 隐空间分布良好。")
        else:
            print(" 暂无隐空间统计数据，请先执行一次训练。")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")

def save_large_image(img_data, path, title=None, is_compare=False):
    """保存大尺寸图片的辅助函数"""
    # 如果是对比图，宽度加倍
    figsize = (8, 4) if is_compare else (4, 4)
    plt.figure(figsize=figsize, dpi=100) # 100 DPI 下 4英寸=400像素
    
    if is_compare:
        # img_data 预期为 (28, 56) 的拼接图
        plt.imshow(img_data, cmap='gray')
    else:
        plt.imshow(img_data, cmap='gray')
        
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def generate_mode():
    """模式 2-1: 生成 9 张大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with torch.no_grad():
        sample = torch.randn(9, latent_channels, latent_spatial_size, latent_spatial_size).to(device)*0.8
        gen_imgs = model.decode(sample).cpu().squeeze().numpy()

        # 1. 保存 9 张独立大图
        for i in range(9):
            # 命名: vae_image_generate_时间戳_编号.png
            filename = IMAGE_DIR / f"vae_image_generate_{timestamp}_{i+1:02d}.png"
            save_large_image(gen_imgs[i], filename)
        
        # 2. 保存 3*3 宫格图
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(gen_imgs[i], cmap='gray')
            ax.axis('off')
        grid_fn = IMAGE_DIR / f"vae_image_generate_{timestamp}.png"
        plt.savefig(grid_fn, bbox_inches='tight')
        plt.close()
        logger.info("Generate 模式运行完毕，图片已保存。")

def compare_mode():
    """模式 2-2: 重构 9 张对比大图和宫格图"""
    if not LATEST_MODEL_PATH.exists():
        logger.error("未找到模型。")
        return

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(datasets.MNIST(root=DATA_DIR, train=False, transform=transform), batch_size=9, shuffle=True)
    orig, _ = next(iter(loader))
    orig = orig.to(device)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with torch.no_grad():
        recon, _, _ = model(orig)
        orig_np = orig.cpu().squeeze().numpy()
        recon_np = recon.cpu().squeeze().numpy()

        # 1. 保存 9 张独立对比大图
        for i in range(9):
            # 左右拼接
            combined = np.hstack((orig_np[i], recon_np[i]))
            filename = IMAGE_DIR / f"vae_image_compare_{timestamp}_{i+1:02d}.png"
            save_large_image(combined, filename, title="Original | Reconstructed", is_compare=True)

        # 2. 保存 3*3 宫格对比总图
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        for i, ax in enumerate(axes.flat):
            combined = np.hstack((orig_np[i], recon_np[i]))
            ax.imshow(combined, cmap='gray')
            ax.set_title(f"Pair {i+1:02d}")
            ax.axis('off')
        grid_fn = IMAGE_DIR / f"vae_image_compare_{timestamp}.png"
        plt.savefig(grid_fn, bbox_inches='tight')
        plt.close()
        logger.info("Compare 模式运行完毕，图片已保存。")

def main():
    while True:
        print("\n" + "="*40)
        print("      VAE 模型管理系统")
        print(" [1] 训练模型 (Train)")
        print(" [2] 测试模式 (Test: Generate/Compare)")
        print(" [3] 查看模型状态 (Status)")
        print(" [0/exit] 退出程序")
        print("="*40)
        
        choice = input("请选择主菜单功能: ").strip().lower()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            while True:
                print("\n  >>> 测试子菜单:")
                print("  [1] Generate (生成 9 张全新数字)")
                print("  [2] Compare  (重构 9 张测试图片对比)")
                print("  [0/exit] 返回主菜单并完全退出")
                
                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == '1':
                    generate_mode()
                elif sub_choice == '2':
                    compare_mode()
                elif sub_choice in ['0', 'exit']:
                    print("退出程序...")
                    sys.exit(0)
                else:
                    print("  无效输入，请输入 1, 2 或 0")
        elif choice == '3':
            show_model_status()
        elif choice in ['0', 'exit']:
            print("退出程序...")
            break
        else:
            print("无效输入，请重新输入 1, 2, 3 或 0")

if __name__ == '__main__':
    main()




