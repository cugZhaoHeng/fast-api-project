import os
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2  # 用于评估模式下的 PHT 裂缝提取

# --- 路径与环境设置 (保留你的原逻辑) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGE_DIR = CURRENT_DIR / "images"
MODEL_DIR = CURRENT_DIR / "models"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. 参数与路径设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 15          # 建议根据算力适当调整
LR = 2e-4
WEIGHT_DECAY = 1e-4

# 扩散模型超参数
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# 图像尺寸
IMAGE_SIZE = 128

# 数据集路径
DFN_DIR = DATA_DIR / "dfn_data"
IMAGES_DIR = DFN_DIR / "images"
CSV_PATH = DFN_DIR / "labels.csv"
MODEL_PATH = MODEL_DIR / "latest_dfn_cddpm.pth"
GEN_DIR = MODEL_DIR / "generated_dfn"
os.makedirs(GEN_DIR, exist_ok=True)

# --- 2. 连续标签归一化与 Dataset 定义 ---

class DFNDataset(Dataset):
    """
    加载 DFN 二维图片和其连续控制标签，并将条件参数归一化到 [-1, 1] 附近
    """
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        
        # 读取并转化为单通道张量，并缩放到 [-1, 1]
        img = Image.open(img_path).convert("L")
        img_np = np.array(img, dtype=np.float32)
        img_tensor = (torch.tensor(img_np) / 127.5) - 1.0  # [0, 255] -> [-1, 1]
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]

        # 提取4个连续参数
        a = row["exponent_a"]
        mu = row["mean_mu"]
        kappa = row["concentration_kappa"]
        n = row["num_fractures"]

        # 归一化公式（将参数大致缩放到 [-1, 1] 区间以便神经网络学习）
        a_norm = (a - 2.25) / 0.75
        mu_norm = mu / 90.0
        kappa_norm = (kappa - 5.5) / 4.5
        n_norm = (n - 15.0) / 10.0

        cond = torch.tensor([a_norm, mu_norm, kappa_norm, n_norm], dtype=torch.float32)
        return img_tensor, cond

# --- 3. 带有连续条件投影的 128x128 U-Net ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU())
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        h = self.transform(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]  # [B, C, 1, 1]
        h = h + time_emb
        h = self.norm2(self.conv2(h))
        return self.relu(h)

class UNet128(nn.Module):
    """适用于 128x128 图像的对称 U-Net，支持多维连续条件输入"""
    def __init__(self):
        super().__init__()
        time_emb_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        
        # 使用 MLP 投影 4 维连续条件
        num_conditions = 4
        self.label_mlp = nn.Sequential(
            nn.Linear(num_conditions, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.down1 = Block(32, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.down3 = Block(128, 256, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot1 = Block(256, 256, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up_block1 = Block(128 + 256, 128, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up_block2 = Block(64 + 128, 64, time_emb_dim)

        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up_block3 = Block(32 + 64, 32, time_emb_dim)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, timestep, label):
        t = self.time_mlp(timestep) + self.label_mlp(label)

        # Down
        x_init = self.conv0(x)         # 128x128, 32
        d1 = self.down1(x_init, t)     # 128x128, 64
        p1 = self.pool(d1)             # 64x64, 64
        d2 = self.down2(p1, t)         # 64x64, 128
        p2 = self.pool(d2)             # 32x32, 128
        d3 = self.down3(p2, t)         # 32x32, 256
        p3 = self.pool(d3)             # 16x16, 256

        # Bottleneck
        bot = self.bot1(p3, t)         # 16x16, 256

        # Up
        u1 = self.up1(bot)             # 32x32, 128
        u1 = torch.cat([u1, d3], dim=1) # 32x32, 128+256
        u1 = self.up_block1(u1, t)     # 32x32, 128

        u2 = self.up2(u1)              # 64x64, 64
        u2 = torch.cat([u2, d2], dim=1) # 64x64, 64+128
        u2 = self.up_block2(u2, t)     # 64x64, 64

        u3 = self.up3(u2)              # 128x128, 32
        u3 = torch.cat([u3, d1], dim=1) # 128x128, 32+64
        u3 = self.up_block3(u3, t)     # 128x128, 32

        return self.out(u3)

# --- 4. 扩散模型调度器 ---

class DDPM(nn.Module):
    def __init__(self, model: UNet128, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward(self, x_0, labels):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = (
            self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        predicted_noise = self.model(x_t, t, labels)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, labels):
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x_t.shape)

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * self.model(x_t, t, labels) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_images, labels):
        self.model.eval()
        img = torch.randn((num_images, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(img, t, i, labels)
        
        # 将其映射回 [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0.0, 1.0)
        return img

# --- 5. 训练主逻辑 ---

def train_model():
    if not os.path.exists(CSV_PATH):
        print("[错误] 未检测到数据集，请先运行第一步代码生成数据集！")
        return

    print("正在加载数据集...")
    dataset = DFNDataset(CSV_PATH, IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    unet = UNet128().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    optimizer = optim.AdamW(ddpm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"检测到历史模型，加载成功。从 Epoch {start_epoch + 1} 继续训练...")

    print(f"开始在设备 [{DEVICE}] 上训练...")
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        ddpm.train()
        avg_loss = 0.0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = ddpm(x, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
        
        print(f"Epoch [{epoch + 1}/{start_epoch + NUM_EPOCHS}] - 平均 MSE 损失: {avg_loss:.5f}")

    torch.save({
        "epoch": start_epoch + NUM_EPOCHS,
        "model_state_dict": ddpm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, MODEL_PATH)
    print(f"模型训练已暂存至: {MODEL_PATH}")

# --- 6. 推理生成与 PHT 霍夫变换验证模式 ---

def load_trained_model():
    unet = UNet128().to(DEVICE)
    ddpm = DDPM(unet, timesteps=TIMESTEPS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.eval()
    return ddpm

def generate_and_verify_mode():
    """
    用户输入一组地质参数条件，模型对应生成50张图，并通过 PHT 霍夫变换自动提取几何信息进行验证。
    """
    if not os.path.exists(MODEL_PATH):
        print("[错误] 未找到已训练模型，请先执行训练。")
        return

    print("\n--- DFN 参数生成与验证系统 ---")
    # 获取输入参数
    target_a = float(input("请输入裂缝长度幂律指数 a (建议 1.5 - 3.0): "))
    target_mu = float(input("请输入走向均值 mu (角度，度，-90.0 - 90.0): "))
    target_kappa = float(input("请输入方向集中度 kappa (建议 1.0 - 10.0): "))
    target_n = int(input("请输入希望每张图生成的裂缝数 N (建议 5 - 25): "))

    # 1. 标签归一化处理 (必须与 Dataset 内公式一致)
    a_norm = (target_a - 2.25) / 0.75
    mu_norm = target_mu / 90.0
    kappa_norm = (target_kappa - 5.5) / 4.5
    n_norm = (target_n - 15.0) / 10.0

    cond_tensor = torch.tensor([a_norm, mu_norm, kappa_norm, n_norm], dtype=torch.float32, device=DEVICE)
    # 扩展 batch 大小为 50 进行统计验证
    num_eval_samples = 50
    cond_batch = cond_tensor.unsqueeze(0).repeat(num_eval_samples, 1)

    print(f"\n正在采样生成 {num_eval_samples} 个 DFN 实例中，请稍后...")
    ddpm = load_trained_model()
    gen_imgs = ddpm.sample(num_eval_samples, cond_batch).cpu().numpy() # [50, 1, 128, 128]

    # 保存一张拼接大图直观展示
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    preview_path = os.path.join(GEN_DIR, "generated_preview.png")
    plt.savefig(preview_path)
    plt.close()
    print(f"生成的网格图像预览已存至: {preview_path}")

    # 2. 利用 PHT (概率霍夫变换) 验证分布特征
    print("\n正在启动 PHT 裂缝几何解译程序...")
    all_lengths = []
    all_angles_deg = []

    for idx in range(num_eval_samples):
        # 取出图像并转化为 OpenCV 的灰度图格式
        img_u8 = (gen_imgs[idx, 0] * 255).astype(np.uint8)
        
        # 二值化翻转（由于霍夫变换常用于检测白线，我们暂时将背景白色255变为0，线条黑色0变为255）
        _, thresh = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY_INV)

        # 执行概率霍夫变换检测直线段
        # 调整这些参数可控制线段检测灵敏度
        lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=15, minLineLength=10, maxLineGap=4)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # 计算角度范围 [-90, 90]
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                # 统一边界
                if angle_deg > 90: angle_deg -= 180
                elif angle_deg < -90: angle_deg += 180
                
                all_lengths.append(length)
                all_angles_deg.append(angle_deg)

    # 3. 统计解译结果
    if len(all_lengths) == 0:
        print("[警告] 未能在生成图像中检测到明显线条，请确认模型是否训练充分！")
        return

    detected_n_avg = len(all_lengths) / num_eval_samples
    detected_mu = np.mean(all_angles_deg)
    detected_std = np.std(all_angles_deg)

    print("\n================== 几何验证报告 ==================")
    print(f" 设定条件 -> 目标密度 N: {target_n:02d}, 目标均值 mu: {target_mu:+.2f}°")
    print(f" 检测结果 -> 平均密度 N: {detected_n_avg:.2f}, 拟合均值 mu: {detected_mu:+.2f}°")
    print(f" 分布发散 -> 标准差 std: {detected_std:.2f}° (目标 kappa: {target_kappa})")
    print("==================================================")
    print("[结论] 若生成的 mu 接近目标 mu，且在 preview 图片上线条走向分布与预期一致，说明 CDDPM 已初步掌握连续参数控制规律。")

# --- 7. 主菜单 ---

def main():
    while True:
        print("\n" + "=" * 40)
        print("    DFN 连续条件扩散模型控制台")
        print(" [1] 开始/继续训练模型 (Train)")
        print(" [2] 输入条件推理与解译验证 (Generate & Verify)")
        print(" [0] 退出程序")
        print("=" * 40)
        choice = input("请选择功能: ").strip()
        if choice == "1":
            train_model()
        elif choice == "2":
            generate_and_verify_mode()
        elif choice == "0":
            break

if __name__ == "__main__":
    main()