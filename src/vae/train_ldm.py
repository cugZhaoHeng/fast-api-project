# -*- coding: utf-8 -*-
"""
3D Diffusion Model for Facies Modeling (Based on VAE) - FIXED VERSION
Author: Your Name
Date: 2025-09-26

This script:
- Loads pre-trained VAE for facies modeling
- Trains a diffusion model in the latent space
- Generates new facies models and saves as txt files
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.logger import create_logger
import math
from tqdm import tqdm

logger = create_logger(__name__)

# -------------------------------
# 配置参数
# -------------------------------
DATA_DIR = r"../../data/npy_files"  # 你的 .npy 文件目录
VAE_MODEL_PATH = "./checkpoints/vae_facies_epoch_100.pth"  # 预训练VAE路径
DIFFUSION_SAVE_DIR = "./diffusion_checkpoints"  # 扩散模型保存路径
GENERATED_DIR = "./generated_diffusion_models"  # 生成模型保存路径

os.makedirs(DIFFUSION_SAVE_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

INPUT_SHAPE = (16, 64, 64)  # 模型尺寸
LATENT_DIM = 64  # 隐空间维度
NUM_CLASSES = 3  # 泥岩、砂岩、流体
BATCH_SIZE = 4  # 3D 数据大，batch 要小
DIFFUSION_EPOCHS = 100
LEARNING_RATE = 1e-4
TIMESTEPS = 1000  # 扩散时间步数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Data dir: {DATA_DIR}")


# -------------------------------
# 1. One-Hot 编码函数 (与VAE相同)
# -------------------------------
def to_onehot(facies_map):
    """
    将原始 facies map (值: 0.1, 10, 200) 转为 one-hot 编码
    输入: (H, W, D)
    输出: (3, H, W, D)
    """
    label_map = np.zeros(facies_map.shape, dtype=np.int64)
    label_map[facies_map == 0.1] = 0  # 泥岩
    label_map[facies_map == 10] = 1  # 砂岩
    label_map[facies_map == 200] = 2  # 流体

    onehot = np.eye(NUM_CLASSES)[label_map]  # -> (H, W, D, 3)
    onehot = np.transpose(onehot, (3, 0, 1, 2))  # -> (3, H, W, D)
    return onehot


# -------------------------------
# 2. 自定义数据集 (与VAE相同)
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
        data_onehot = to_onehot(data)  # (3, 16, 64, 64)
        return torch.from_numpy(data_onehot).float()


# -------------------------------
# 3. VAE 模型 (与之前相同)
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
# 4. 扩散模型组件 - FIXED VERSION
# -------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""

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


class SimpleMLPDiffusion(nn.Module):
    """简单的MLP扩散模型，直接在潜在空间操作"""

    def __init__(self, latent_dim=LATENT_DIM, time_emb_dim=128, hidden_dims=[512, 256, 128]):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 构建MLP层
        layers = []
        input_dim = latent_dim + time_emb_dim  # 潜在向量 + 时间嵌入

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, time):
        # x shape: [batch_size, latent_dim]
        # time shape: [batch_size]

        # 时间嵌入
        t_emb = self.time_mlp(time)  # [batch_size, time_emb_dim]

        # 拼接输入和时间嵌入
        x = torch.cat([x, t_emb], dim=-1)  # [batch_size, latent_dim + time_emb_dim]

        # 通过MLP
        return self.mlp(x)


class DiffusionModel:
    """扩散模型主类"""

    def __init__(self, vae_model, timesteps=TIMESTEPS):
        self.timesteps = timesteps
        self.vae = vae_model
        self.vae.eval()  # 冻结VAE

        # 定义beta调度（线性调度，更稳定）
        self.betas = self._linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 扩散模型 - 使用简单的MLP
        self.model = SimpleMLPDiffusion().to(DEVICE)

    def _linear_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """线性beta调度，更稳定"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def extract(self, a, t, x_shape):
        """从张量a中提取对应时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t):
        """计算损失 - 简化版本"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        return nn.functional.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """从p_theta(x_{t-1} | x_t)采样"""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(torch.sqrt(1.0 / self.alphas), t, x.shape)

        # 使用模型预测噪声
        predicted_noise = self.model(x, t)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.betas, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size=1, save_intermediate=False):
        """从扩散模型生成样本"""
        self.model.eval()

        # 从纯噪声开始
        shape = (batch_size, LATENT_DIM)
        x = torch.randn(shape, device=DEVICE)

        intermediate_samples = []

        if save_intermediate:
            # 保存初始噪声状态
            intermediate_samples.append(x.clone())

        for i in tqdm(reversed(range(0, self.timesteps)), desc='采样进度', total=self.timesteps):
            t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            x = self.p_sample(x, t, i)

            # 保存关键步骤的中间结果
            if save_intermediate and i in [self.timesteps - 1, self.timesteps // 2, 1]:
                intermediate_samples.append(x.clone())

        if save_intermediate:
            # 保存最终结果
            intermediate_samples.append(x.clone())
            return x, intermediate_samples
        else:
            return x


# -------------------------------
# 5. 保存训练损失曲线
# -------------------------------
def save_training_loss(loss_history, save_path="diffusion_training_loss.png"):
    """保存训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Diffusion Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training loss curve saved to {save_path}")


# -------------------------------
# 6. 保存去噪过程样本
# -------------------------------
def save_denoising_samples(intermediate_samples, vae_model, save_dir="denoising_process"):
    """保存去噪过程中的关键步骤样本"""
    os.makedirs(save_dir, exist_ok=True)

    step_names = ["initial_noise", "first_denoise", "mid_denoise", "final_denoise"]

    for i, z in enumerate(intermediate_samples):
        with torch.no_grad():
            # 使用VAE解码
            prob_map = vae_model.decode(z)  # (1, 3, 16, 64, 64)

            # 取最大概率类别
            class_map = torch.argmax(prob_map, dim=1).cpu().squeeze().numpy()  # (16, 64, 64)

            # 映射回原始值
            k_map = np.zeros_like(class_map, dtype=np.float32)
            k_map[class_map == 0] = 0.1  # 泥岩
            k_map[class_map == 1] = 10  # 砂岩
            k_map[class_map == 2] = 200  # 流体

            # 保存为txt文件
            txt_path = os.path.join(save_dir, f"{step_names[i]}.txt")
            save_flat_data_as_txt(k_map, txt_path)

            print(f"Saved {step_names[i]} to {txt_path}")


def save_flat_data_as_txt(data, filename):
    """将数据展平并保存为txt文件，每行一个数据"""
    flat_data = data.flatten()
    with open(filename, 'w') as f:
        for value in flat_data:
            f.write(f"{value}\n")


# -------------------------------
# 7. 扩散模型训练函数
# -------------------------------
def train_diffusion():
    """训练扩散模型"""

    # 加载数据集
    dataset = FaciesDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} models for diffusion training.")

    # 加载预训练VAE
    vae_model = VAE3D_Facies().to(DEVICE)
    if os.path.exists(VAE_MODEL_PATH):
        vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        print("✅ VAE model loaded successfully.")
    else:
        print("❌ VAE model not found. Please train VAE first.")
        return

    # 初始化扩散模型
    diffusion = DiffusionModel(vae_model)
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting Diffusion Model Training...")

    loss_history = []

    for epoch in range(1, DIFFUSION_EPOCHS + 1):
        diffusion.model.train()
        total_loss = 0

        for batch_idx, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}/{DIFFUSION_EPOCHS}')):
            data = data.to(DEVICE)  # (B, 3, 16, 64, 64)

            # 使用VAE编码到潜在空间
            with torch.no_grad():
                mu, log_var = vae_model.encode(data)
                latent_data = vae_model.reparameterize(mu, log_var)  # (B, latent_dim)

            # 采样时间步
            t = torch.randint(0, diffusion.timesteps, (data.shape[0],), device=DEVICE).long()

            # 计算损失
            loss = diffusion.p_losses(latent_data, t)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')

        # 每10个epoch保存一次模型和损失曲线
        if epoch % 10 == 0:
            model_path = os.path.join(DIFFUSION_SAVE_DIR, f"diffusion_epoch_{epoch}.pth")
            torch.save(diffusion.model.state_dict(), model_path)
            print(f"Diffusion model saved to {model_path}")

            # 保存损失曲线
            loss_curve_path = os.path.join(DIFFUSION_SAVE_DIR, f'training_loss_epoch_{epoch}.png')
            save_training_loss(loss_history, loss_curve_path)

            # 生成并保存去噪过程样本
            print("Generating denoising process samples...")
            _, intermediate_samples = diffusion.sample(batch_size=1, save_intermediate=True)
            denoising_dir = os.path.join(DIFFUSION_SAVE_DIR, f"denoising_process_epoch_{epoch}")
            save_denoising_samples(intermediate_samples, vae_model, denoising_dir)

    # 保存最终模型和损失曲线
    final_path = os.path.join(DIFFUSION_SAVE_DIR, "diffusion_final.pth")
    torch.save(diffusion.model.state_dict(), final_path)

    # 保存最终损失曲线
    final_loss_path = os.path.join(DIFFUSION_SAVE_DIR, "final_training_loss.png")
    save_training_loss(loss_history, final_loss_path)

    print(f"✅ Final diffusion model saved to {final_path}")
    print(f"✅ Final training loss curve saved to {final_loss_path}")

    return diffusion


# -------------------------------
# 8. 生成和保存函数
# -------------------------------
def generate_facies_from_diffusion(diffusion_model, vae_model, num_models=10):
    """使用扩散模型生成新的相模型"""

    print(f"🔄 Generating {num_models} new facies models...")

    for i in range(num_models):
        # 从扩散模型生成潜在向量
        with torch.no_grad():
            latent_z = diffusion_model.sample(batch_size=1)  # (1, latent_dim)

            # 使用VAE解码
            prob_map = vae_model.decode(latent_z)  # (1, 3, 16, 64, 64)

            # 取最大概率类别
            class_map = torch.argmax(prob_map, dim=1).cpu().squeeze().numpy()  # (16, 64, 64)

            # 映射回原始值
            k_map = np.zeros_like(class_map, dtype=np.float32)
            k_map[class_map == 0] = 0.1  # 泥岩
            k_map[class_map == 1] = 10  # 砂岩
            k_map[class_map == 2] = 200  # 流体

            # 保存为txt文件
            txt_path = os.path.join(GENERATED_DIR, f"diffusion_model_{i + 1:03d}.txt")
            save_model_as_txt(k_map, txt_path)

            # 同时保存为npy文件
            npy_path = os.path.join(GENERATED_DIR, f"diffusion_model_{i + 1:03d}.npy")
            np.save(npy_path, k_map)

            print(f"Generated model {i + 1}/{num_models}: {txt_path}")


def save_model_as_txt(model_data, filename):
    """将3D模型数据保存为txt文件"""
    H, W, D = model_data.shape

    with open(filename, 'w') as f:
        # 写入维度信息
        f.write(f"Dimensions: {H} {W} {D}\n")
        f.write("Data:\n")

        # 按照z, y, x的顺序写入数据（根据tNavigator的要求调整）
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    value = model_data[y, x, z]
                    f.write(f"{value:.6f}\n")

    print(f"✅ Model saved as TXT: {filename}")


def visualize_generated_model(model_data, slice_idx=8, title="Generated Facies Model"):
    """可视化生成的模型"""
    plt.figure(figsize=(10, 8))
    plt.imshow(model_data[slice_idx, :, :], cmap='viridis', vmin=0, vmax=250)
    plt.colorbar(label='Facies Value\n(0.1=Mud, 10=Sand, 200=Fluid)')
    plt.title(f'{title} (Z-Slice {slice_idx})')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 添加图例说明
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='darkblue', alpha=0.7, label='Mud (0.1)'),
        plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.7, label='Sand (10)'),
        plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.7, label='Fluid (200)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(GENERATED_DIR, f"{title.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------------
# 9. 测试函数
# -------------------------------
def test_diffusion():
    """测试训练好的扩散模型"""

    # 加载VAE模型
    vae_model = VAE3D_Facies().to(DEVICE)
    if os.path.exists(VAE_MODEL_PATH):
        vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        print("✅ VAE model loaded successfully.")
    else:
        print("❌ VAE model not found.")
        return

    # 加载扩散模型
    diffusion_model = DiffusionModel(vae_model)
    diffusion_path = os.path.join(DIFFUSION_SAVE_DIR, "diffusion_final.pth")

    if os.path.exists(diffusion_path):
        diffusion_model.model.load_state_dict(torch.load(diffusion_path, map_location=DEVICE))
        print("✅ Diffusion model loaded successfully.")
    else:
        print("❌ Diffusion model not found. Please train first.")
        return

    # 生成模型
    generate_facies_from_diffusion(diffusion_model, vae_model, num_models=20)

    # 生成并保存去噪过程样本
    print("Generating denoising process samples for visualization...")
    _, intermediate_samples = diffusion_model.sample(batch_size=1, save_intermediate=True)
    denoising_dir = os.path.join(GENERATED_DIR, "denoising_process")
    save_denoising_samples(intermediate_samples, vae_model, denoising_dir)

    # 可视化一个生成的模型
    sample_model_path = os.path.join(GENERATED_DIR, "diffusion_model_001.npy")
    if os.path.exists(sample_model_path):
        sample_model = np.load(sample_model_path)
        visualize_generated_model(sample_model, title="Diffusion Generated Facies Model")

    print("✅ Diffusion model testing completed!")


# -------------------------------
# 10. 主函数
# -------------------------------
if __name__ == "__main__":
    print("\n🎯 3D Diffusion Model for Facies Modeling - FIXED VERSION\n")

    # 选择运行模式
    mode = input("Enter mode ('train' to train diffusion, 'test' to generate): ").strip().lower()

    if mode == "train":
        trained_diffusion = train_diffusion()
        # 训练后立即测试生成一些样本
        test_diffusion()
    elif mode == "test":
        test_diffusion()
    else:
        print("Invalid mode. Use 'train' or 'test'.")