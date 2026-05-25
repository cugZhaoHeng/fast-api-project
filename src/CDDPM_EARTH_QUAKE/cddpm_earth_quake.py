import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 超参数与类别数量设定 (严格对照论文)
# ==========================================
SEQ_LEN = 6400       # 序列标准化长度
TIMESTEPS = 1000     # 扩散总步数 T
BETA_START = 1e-4    # β1
BETA_END = 0.02      # βT
LR = 0.0002          # 学习率
BATCH_SIZE = 16

# 论文中 4 个条件的类别数估计 (根据论文2.Data部分)
NUM_VS30 = 2    # NEHRP B/C, NEHRP D
NUM_F = 3       # Normal, Reverse, Strike-slip
NUM_MW = 5      # 比如 5.5, 6, 6.5, 7, 7.5
NUM_RRUP = 7    # 比如 5, 10, 20, 40, 60, 80, 100

EMB_DIM = 64    # 隐层向量维度 d_model

# ==========================================
# 2. 核心创新模块：多标签条件嵌入 (Multi-label Embedding)
# ==========================================
def get_positional_encoding(pos, d_model):
    """标准的正弦位置编码生成，对应论文 Equation (8) 和 (9)"""
    pe = torch.zeros(d_model)
    position = torch.tensor([pos]).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[0::2] = torch.sin(position * div_term)
    pe[1::2] = torch.cos(position * div_term)
    return pe

class MultiLabelConditionEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # nn.Embedding 在计算上等效于 One-hot + Linear 层映射
        self.emb_vs30 = nn.Embedding(NUM_VS30, d_model)
        self.emb_f = nn.Embedding(NUM_F, d_model)
        self.emb_mw = nn.Embedding(NUM_MW, d_model)
        self.emb_rrup = nn.Embedding(NUM_RRUP, d_model)

        # 预先生成层级位置编码 (论文设定：VS30属于Level 1, 其他属于Level 2/3)
        self.register_buffer('pe_level1', get_positional_encoding(1, d_model))
        self.register_buffer('pe_level2', get_positional_encoding(2, d_model))
        self.register_buffer('pe_level3', get_positional_encoding(3, d_model))

        # 将4个条件合并后的融合网络
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, vs30, f, mw, rrup):
        # 1. 取出 Embedding (相当于 One-hot 映射)
        e_vs30 = self.emb_vs30(vs30)
        e_f = self.emb_f(f)
        e_mw = self.emb_mw(mw)
        e_rrup = self.emb_rrup(rrup)

        # 2. 注入层级位置编码 (Add Positional Encoding)
        e_vs30 = e_vs30 + self.pe_level1
        e_f = e_f + self.pe_level2
        e_mw = e_mw + self.pe_level3
        e_rrup = e_rrup + self.pe_level3

        # 3. 拼接并融合 (Concatenation)
        # shape: [B, d_model*4]
        concat_emb = torch.cat([e_vs30, e_f, e_mw, e_rrup], dim=-1)
        # shape: [B, d_model]
        cond_emb = self.fusion_mlp(concat_emb) 
        return cond_emb

# ==========================================
# 3. 1D U-Net 基础构件与网络定义
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    """处理扩散步数 t 的时间位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_cond_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        # 注入时间+条件的投影层
        self.proj_cond = nn.Linear(time_cond_dim, out_ch)

    def forward(self, x, cond):
        h = self.conv(x)
        h = self.norm(h)
        # cond shape: [B, time_cond_dim] -> [B, out_ch, 1] 广播到1D特征图上
        cond = self.proj_cond(cond).unsqueeze(-1)
        return self.act(h + cond)

class UNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        # 时间与标签条件网络
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(EMB_DIM),
            nn.Linear(EMB_DIM, EMB_DIM * 4),
            nn.SiLU(),
            nn.Linear(EMB_DIM * 4, EMB_DIM)
        )
        self.label_cond_net = MultiLabelConditionEmbedding(EMB_DIM)
        
        cond_dim = EMB_DIM  # 时间与标签相加

        # U-Net 层
        self.inc = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.down1 = ConvBlock(32, 64, cond_dim)
        self.pool1 = nn.MaxPool1d(2) # 6400 -> 3200
        self.down2 = ConvBlock(64, 128, cond_dim)
        self.pool2 = nn.MaxPool1d(2) # 3200 -> 1600
        
        self.mid1 = ConvBlock(128, 128, cond_dim)
        
        self.up1 = nn.Upsample(scale_factor=2) # 1600 -> 3200
        self.up_conv1 = ConvBlock(128 + 64, 64, cond_dim)
        self.up2 = nn.Upsample(scale_factor=2) # 3200 -> 6400
        self.up_conv2 = ConvBlock(64 + 32, 32, cond_dim)
        
        self.outc = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x, t, vs30, f, mw, rrup):
        # 处理时间 t
        t_emb = self.time_mlp(t)
        # 处理多标签物理条件
        label_emb = self.label_cond_net(vs30, f, mw, rrup)
        # 核心：将时间步 t 和 物理条件 C 进行联合 (这里使用相加法)
        cond = t_emb + label_emb 

        # Forward U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1, cond)
        p2 = self.pool1(x2)
        x3 = self.down2(p2, cond)
        p3 = self.pool2(x3)
        
        mid = self.mid1(p3, cond)
        
        u1 = self.up1(mid)
        u1 = torch.cat([u1, x3], dim=1) # Skip connection
        u1 = self.up_conv1(u1, cond)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up_conv2(u2, cond)
        
        out = self.outc(u2)
        return out

# ==========================================
# 4. DDPM 类与加噪/去噪逻辑
# ==========================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_timesteps = TIMESTEPS
        
        # 论文设定：线性增加的 variance schedule (β_1 = 10^-4 to β_T = 0.02)
        betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """前向过程：向真实数据添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x_start, t, vs30, f, mw, rrup):
        """计算损失：预测噪声的 MSE Loss (Equation 7)"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, vs30, f, mw, rrup)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x, t, vs30, f, mw, rrup):
        """反向过程的单步去噪"""
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))

        # 让模型预测出加入的噪声
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, vs30, f, mw, rrup) / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0].item() == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t # 简化的方差设定
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape, vs30, f, mw, rrup, device):
        """生成完整波形的推理过程"""
        x = torch.randn(shape, device=device) # 初始化纯噪声 x_T
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, vs30, f, mw, rrup)
        return x

# ==========================================
# 5. 模拟数据集构建 (Mock Data)
# ==========================================
class MockEarthquakeDataset(Dataset):
    """
    因为没有真实的 PEER 地震波数据，这里生成随机的时间序列信号
    以及随机分配的 4 个物理标签来跑通代码。
    """
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 伪造一个 1D 波形信号，加入了正弦震荡衰减模拟地震波
        time_seq = torch.linspace(0, 10, SEQ_LEN)
        wave = torch.sin(time_seq * 5) * torch.exp(-time_seq/2) + torch.randn(SEQ_LEN)*0.05
        wave = wave.unsqueeze(0) # shape: [1, 6400]

        # 随机分配多标签
        vs30 = torch.randint(0, NUM_VS30, (1,)).item()
        f = torch.randint(0, NUM_F, (1,)).item()
        mw = torch.randint(0, NUM_MW, (1,)).item()
        rrup = torch.randint(0, NUM_RRUP, (1,)).item()

        return wave, vs30, f, mw, rrup

# ==========================================
# 6. 主程序：训练与推理
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 初始化模型
    unet = UNet1D()
    diffusion_model = GaussianDiffusion(unet).to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LR)

    # 2. 准备数据
    dataset = MockEarthquakeDataset(size=100) # 模拟100条记录
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 开始训练
    epochs = 5 # 演示用途只跑5轮
    print("--- 开启训练 ---")
    diffusion_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (waves, vs30, f, mw, rrup) in enumerate(dataloader):
            waves = waves.to(device)
            vs30, f, mw, rrup = vs30.to(device), f.to(device), mw.to(device), rrup.to(device)

            # 随机采样时间步 t
            t = torch.randint(0, TIMESTEPS, (waves.shape[0],), device=device).long()

            # 计算 Loss 并反向传播
            optimizer.zero_grad()
            loss = diffusion_model.p_losses(waves, t, vs30, f, mw, rrup)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # 4. 推理过程 (Inference / Generation)
    print("--- 开始推理模拟 ---")
    diffusion_model.eval()
    
    # 我们设定一个工程场景：NEHRP D类场地(1), Normal断层(0), 震级6.0(1), 距离10km(1)
    cond_vs30 = torch.tensor([1]).to(device)
    cond_f    = torch.tensor([0]).to(device)
    cond_mw   = torch.tensor([1]).to(device)
    cond_rrup = torch.tensor([1]).to(device)

    # 生成波形
    shape = (1, 1, SEQ_LEN) # [Batch, Channels, Length]
    generated_wave = diffusion_model.sample(shape, cond_vs30, cond_f, cond_mw, cond_rrup, device)
    
    # 5. 可视化
    generated_wave = generated_wave.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(generated_wave, label="Generated Seismic Wave", linewidth=0.5)
    plt.title("ML-cDDPM Generated Ground Motion (Scenario: D, Normal, Mw=6.0, R=10km)")
    plt.xlabel("Time steps")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("生成完成！图片已展示。")