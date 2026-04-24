import os
import math
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import lasio
except ImportError:
    lasio = None


# =========================
# 1. 配置
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "log.csv"   # 改成你的 csv 或 las 路径
IS_LAS = DATA_PATH.lower().endswith(".las")

LOG_COLUMNS_CANDIDATES = {
    "GR": ["GR", "GAMMA", "GAMMA_RAY"],
    "RHOB": ["RHOB", "RHOZ", "RHO_B"],
    "NPHI": ["NPHI", "NPHI_LS", "NPHI_N", "CNLS", "CNPOR", "NPLS"],
    "RILD": ["RILD", "RLLD", "ILD", "RT"]
}

WINDOW_SIZE = 128
WINDOW_STRIDE = 32
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

NUM_WORKERS = 0


# =========================
# 2. 工具函数
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    upper_map = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in upper_map:
            return upper_map[name.upper()]
    return None


def load_csv_or_las(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        return df

    if path.lower().endswith(".las"):
        if lasio is None:
            raise ImportError("读取 LAS 需要安装 lasio：pip install lasio")
        las = lasio.read(path)
        df = las.df().reset_index()
        return df

    raise ValueError("仅支持 .csv 或 .las 文件")


def select_required_logs(df: pd.DataFrame) -> pd.DataFrame:
    selected = {}
    for std_name, candidates in LOG_COLUMNS_CANDIDATES.items():
        matched = find_matching_column(df, candidates)
        if matched is not None:
            selected[std_name] = pd.to_numeric(df[matched], errors="coerce")
    print("实际匹配到的列：", selected.keys())
    missing = [k for k in ["GR", "RHOB", "NPHI", "RILD"] if k not in selected]
    if missing:
        raise ValueError(f"缺少必要曲线列: {missing}，当前可识别列: {list(df.columns)}")

    out = pd.DataFrame(selected)

    # 常见非法值处理，可按数据集再调
    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))
    out["NPHI"] = out["NPHI"] / 100.0
    out["NPHI"] = out["NPHI"].mask((out["NPHI"] < -0.15) | (out["NPHI"] > 1.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)

    return out


def preprocess_logs(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    # 电阻率先做 log10
    df = df.copy()
    df["RILD"] = np.log10(df["RILD"])

    # 缺失值简单插值
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.dropna().reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"有效样本太少，清洗后仅 {len(df)} 行，小于窗口长度 {WINDOW_SIZE}")

    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values.astype(np.float32))
    return arr, scaler


def make_windows(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(arr) - window_size + 1, stride):
        w = arr[start:start + window_size]   # [L, C]
        print(f"w length: {len(w)}")
        windows.append(w)
    windows = np.stack(windows, axis=0)     # [N, L, C]
    windows = np.transpose(windows, (0, 2, 1))  # [N, C, L]
    return windows.astype(np.float32)


# =========================
# 3. 数据集
# =========================
class WellLogDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


# =========================
# 4. 时间步编码
# =========================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# =========================
# 5. 最小 1D DDPM 去噪网络
# =========================
class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_term = self.time_mlp(t_emb).unsqueeze(-1)
        h = h + time_term
        h = self.block2(h)
        return x + h


class SimpleDenoiser1D(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.res1 = ResidualBlock1D(hidden_channels, time_dim)
        self.res2 = ResidualBlock1D(hidden_channels, time_dim)
        self.res3 = ResidualBlock1D(hidden_channels, time_dim)

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(x)
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        h = self.res3(h, t_emb)
        out = self.output_proj(h)
        return out


# =========================
# 6. DDPM
# =========================
class DDPM:
    def __init__(self, model: nn.Module, timesteps: int = 200,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_omb * noise
        return xt, noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.model(xt, t)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        batch_size = x.shape[0]
        t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        pred_noise = self.model(x, t_batch)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x


# =========================
# 7. 训练与可视化
# =========================
def train(ddpm: DDPM, dataloader: DataLoader, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=lr)

    ddpm.model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (batch.size(0),), device=ddpm.device).long()

            loss = ddpm.p_losses(batch, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")


def inverse_transform_sample(sample: np.ndarray, scaler: StandardScaler) -> pd.DataFrame:
    # sample: [C, L] -> [L, C]
    arr = np.transpose(sample, (1, 0))
    arr = scaler.inverse_transform(arr)

    df = pd.DataFrame(arr, columns=["GR", "RHOB", "NPHI", "RILD_log10"])
    df["RILD"] = 10 ** df["RILD_log10"]
    df = df.drop(columns=["RILD_log10"])
    return df


def plot_generated_sample(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    depth = np.arange(len(df))

    fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

    axes[0].plot(df["GR"], depth)
    axes[0].set_title("GR")

    axes[1].plot(df["RHOB"], depth)
    axes[1].set_title("RHOB")

    axes[2].plot(df["NPHI"], depth)
    axes[2].set_title("NPHI")

    axes[3].plot(df["RILD"], depth)
    axes[3].set_title("RILD")

    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Generated well-log window")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =========================
# 8. 主流程
# =========================
def main() -> None:
    set_seed(SEED)

    print("Loading data...")
    raw_df = load_csv_or_las(DATA_PATH)
    logs_df = select_required_logs(raw_df)
    arr, scaler = preprocess_logs(logs_df)

    print(f"Cleaned data shape: {arr.shape}")  # [N_depth, 4]

    windows = make_windows(arr, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"Windows shape: {windows.shape}")   # [N_windows, 4, 128]

    # dataset = WellLogDataset(windows)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    #     drop_last=True
    # )
    #
    # model = SimpleDenoiser1D(in_channels=4, hidden_channels=64, time_dim=128)
    # ddpm = DDPM(
    #     model=model,
    #     timesteps=DIFFUSION_STEPS,
    #     beta_start=BETA_START,
    #     beta_end=BETA_END,
    #     device=DEVICE
    # )
    #
    # print(f"Training on {DEVICE}...")
    # train(ddpm, dataloader, epochs=EPOCHS, lr=LR)
    #
    # print("Sampling...")
    # ddpm.model.eval()
    # samples = ddpm.sample((1, 4, WINDOW_SIZE))  # [B, C, L]
    # sample_np = samples[0].detach().cpu().numpy()
    #
    # sample_df = inverse_transform_sample(sample_np, scaler)
    # print(sample_df.head())
    #
    # plot_generated_sample(sample_df, save_path="generated_well_log.png")
    # print("Done. Saved figure to generated_well_log.png")


if __name__ == "__main__":
    main()