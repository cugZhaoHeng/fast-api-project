import os
import math
from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import lasio
except ImportError:
    lasio = None

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# =========================
# 1. 配置
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = str(DATA_DIR / "well_log_data" / "log.csv")   # 改成你的 csv 或 las 路径
WINDOW_SIZE = 128
WINDOW_STRIDE = 32

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

# 人工目标缺失段长度，单位是采样点
TARGET_MIN_LEN = 16
TARGET_MAX_LEN = 48

# 每次随机遮挡多少条曲线作为 target
MIN_TARGET_CHANNELS = 1
MAX_TARGET_CHANNELS = 2

LOG_COLUMNS_CANDIDATES = {
    "GR": ["GR", "GAMMA", "GAMMA_RAY"],
    "RHOB": ["RHOB", "RHOZ", "RHO_B"],
    "NPHI": ["NPHI", "NPHI_LS", "NPHI_N", "CNLS", "CNPOR", "NPLS"],
    "RILD": ["RILD", "RLLD", "ILD", "RT"]
}

LOG_NAMES = ["GR", "RHOB", "NPHI", "RILD"]


# =========================
# 2. 工具函数
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def find_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    upper_map = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in upper_map:
            return upper_map[name.upper()]
    return None


def load_csv_or_las(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)

    if path.lower().endswith(".las"):
        if lasio is None:
            raise ImportError("读取 LAS 需要安装 lasio：pip install lasio")
        las = lasio.read(path)
        return las.df().reset_index()

    raise ValueError("仅支持 .csv 或 .las 文件")


def select_required_logs(df: pd.DataFrame) -> pd.DataFrame:
    selected = {}
    matched_map = {}

    for std_name, candidates in LOG_COLUMNS_CANDIDATES.items():
        matched = find_matching_column(df, candidates)
        if matched is not None:
            selected[std_name] = pd.to_numeric(df[matched], errors="coerce")
            matched_map[std_name] = matched

    print("列匹配结果:", matched_map)

    missing = [k for k in LOG_NAMES if k not in selected]
    if missing:
        raise ValueError(f"缺少必要曲线列: {missing}，当前列: {list(df.columns)}")

    out = pd.DataFrame(selected)

    # 基础异常值处理
    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))

    # NPHI 有些数据是百分数，例如 25，需要除以 100
    if out["NPHI"].dropna().median() > 1.5:
        out["NPHI"] = out["NPHI"] / 100.0

    out["NPHI"] = out["NPHI"].mask((out["NPHI"] < -0.2) | (out["NPHI"] > 1.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)

    return out


def preprocess_logs_keep_missing(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, SimpleStandardScaler]:
    """
    返回：
    x_scaled_filled: [N, C]，缺失填 0 后的标准化数据
    observed_mask:   [N, C]，1=原始有值，0=原始缺失
    scaler
    """
    df = df.copy()

    # 电阻率 log10
    df["RILD"] = np.log10(df["RILD"])

    values = df[LOG_NAMES].values.astype(np.float32)
    observed_mask = np.isfinite(values).astype(np.float32)

    scaler = SimpleStandardScaler()
    x_scaled = scaler.fit_transform(values)

    # 缺失位置填 0。因为标准化后 0 约等于均值。
    x_scaled_filled = np.where(np.isfinite(x_scaled), x_scaled, 0.0).astype(np.float32)

    return x_scaled_filled, observed_mask.astype(np.float32), scaler


def make_windows_with_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    window_size: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_windows = []
    m_windows = []

    for start in range(0, len(arr) - window_size + 1, stride):
        xw = arr[start:start + window_size]      # [L, C]
        mw = mask[start:start + window_size]     # [L, C]

        # 如果这个窗口观测值太少，跳过
        if mw.mean() < 0.7:
            continue

        x_windows.append(xw)
        m_windows.append(mw)

    x_windows = np.stack(x_windows, axis=0)      # [N, L, C]
    m_windows = np.stack(m_windows, axis=0)

    x_windows = np.transpose(x_windows, (0, 2, 1))  # [N, C, L]
    m_windows = np.transpose(m_windows, (0, 2, 1))  # [N, C, L]

    return x_windows.astype(np.float32), m_windows.astype(np.float32)


# =========================
# 3. Dataset：动态生成人工 target mask
# =========================
class WellLogCDDPMDataset(Dataset):
    def __init__(
        self,
        x_windows: np.ndarray,
        obs_masks: np.ndarray,
        target_min_len: int = 16,
        target_max_len: int = 48,
        min_target_channels: int = 1,
        max_target_channels: int = 2
    ):
        self.x = torch.from_numpy(x_windows)          # [N, C, L]
        self.m_obs = torch.from_numpy(obs_masks)      # [N, C, L]
        self.target_min_len = target_min_len
        self.target_max_len = target_max_len
        self.min_target_channels = min_target_channels
        self.max_target_channels = max_target_channels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x0 = self.x[idx].clone()
        m_obs = self.m_obs[idx].clone()

        c, l = x0.shape

        # 人工 target mask：只允许从真实观测区域里挖
        m_ta = torch.zeros_like(x0)

        num_ch = random.randint(self.min_target_channels, self.max_target_channels)
        channels = random.sample(range(c), k=num_ch)

        for ch in channels:
            seg_len = random.randint(self.target_min_len, self.target_max_len)
            if seg_len >= l:
                seg_len = l // 2

            # 多尝试几次，尽量找到一段原始观测比较完整的位置
            found = False
            for _ in range(20):
                start = random.randint(0, l - seg_len)
                end = start + seg_len
                if m_obs[ch, start:end].mean() > 0.95:
                    m_ta[ch, start:end] = 1.0
                    found = True
                    break

            # 找不到就退化为随机可观测点附近
            if not found:
                valid_idx = torch.where(m_obs[ch] > 0.5)[0]
                if len(valid_idx) > seg_len:
                    start_pos = random.randint(0, len(valid_idx) - seg_len)
                    chosen = valid_idx[start_pos:start_pos + seg_len]
                    m_ta[ch, chosen] = 1.0

        # 条件 mask：
        # m_obs = 1 的地方是真实可见
        # m_ta = 1 的地方是人工挖掉
        # m_cond = 1 表示模型可见；0 表示不可见
        m_cond = m_obs * (1.0 - m_ta)

        x_cond = x0 * m_cond

        return {
            "x0": x0,
            "m_obs": m_obs,
            "m_ta": m_ta,
            "m_cond": m_cond,
            "x_cond": x_cond
        }


# =========================
# 4. Time embedding
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
# 5. CDDPM 去噪网络
# =========================
class ConditionalResidualBlock1D(nn.Module):
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
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.block2(h)
        return x + h


class ConditionalDenoiser1D(nn.Module):
    """
    输入：
    x_ta_t: [B, C, L] 目标区域的带噪数据，非目标区域为 0
    x_cond: [B, C, L] 条件数据，target/missing 区域为 0
    m_cond: [B, C, L] 条件 mask，1=可见，0=不可见
    t:      [B]

    输出：
    pred_noise: [B, C, L]
    """
    def __init__(self, log_channels: int = 4, hidden_channels: int = 64, time_dim: int = 128):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 拼接输入：
        # x_ta_t  C
        # x_cond  C
        # m_cond  C
        # 共 3C
        in_channels = log_channels * 3

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.res1 = ConditionalResidualBlock1D(hidden_channels, time_dim)
        self.res2 = ConditionalResidualBlock1D(hidden_channels, time_dim)
        self.res3 = ConditionalResidualBlock1D(hidden_channels, time_dim)
        self.res4 = ConditionalResidualBlock1D(hidden_channels, time_dim)

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, log_channels, kernel_size=1)
        )

    def forward(
        self,
        x_ta_t: torch.Tensor,
        x_cond: torch.Tensor,
        m_cond: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)

        x = torch.cat([x_ta_t, x_cond, m_cond], dim=1)

        h = self.input_proj(x)
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        h = self.res3(h, t_emb)
        h = self.res4(h, t_emb)

        return self.output_proj(h)


# =========================
# 6. CDDPM
# =========================
class CDDPM:
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
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

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        xt = sqrt_ab * x0 + sqrt_omb * noise
        return xt, noise

    def p_losses(self, batch: dict) -> torch.Tensor:
        x0 = batch["x0"].to(self.device)
        x_cond = batch["x_cond"].to(self.device)
        m_cond = batch["m_cond"].to(self.device)
        m_ta = batch["m_ta"].to(self.device)

        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()

        xt, noise = self.q_sample(x0, t)

        # 只把人工目标区域的 noisy 数据作为 target 输入
        x_ta_t = xt * m_ta

        pred_noise = self.model(x_ta_t, x_cond, m_cond, t)

        # loss 只在人工 target 区域计算
        denom = m_ta.sum().clamp(min=1.0)
        loss = (((pred_noise - noise) ** 2) * m_ta).sum() / denom

        return loss

    @torch.no_grad()
    def impute(
        self,
        x0_filled: torch.Tensor,
        m_obs: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        补全函数。

        x0_filled: [B, C, L]，缺失位置已经填 0
        m_obs:     [B, C, L]，1=真实观测，0=真实缺失
        target_mask:
            None：补真实缺失区域，即 1 - m_obs
            非 None：补指定区域，例如人工遮挡区域

        返回：
            x_completed: [B, C, L]
        """
        self.model.eval()

        x0_filled = x0_filled.to(self.device)
        m_obs = m_obs.to(self.device)

        if target_mask is None:
            m_target = 1.0 - m_obs
        else:
            m_target = target_mask.to(self.device)

        m_cond = 1.0 - m_target
        x_cond = x0_filled * m_cond

        # 从纯噪声开始生成 target 区域
        x = torch.randn_like(x0_filled) * m_target

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)

            pred_noise = self.model(x, x_cond, m_cond, t)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )

            if step > 0:
                noise = torch.randn_like(x)
                x_prev = mean + torch.sqrt(beta_t) * noise
            else:
                x_prev = mean

            # 只更新 target 区域，条件区域始终保持 0
            x = x_prev * m_target

        x_completed = x_cond + x * m_target
        return x_completed


# =========================
# 7. 训练
# =========================
def train(cddpm: CDDPM, dataloader: DataLoader, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(cddpm.model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        cddpm.model.train()
        total_loss = 0.0

        for batch in dataloader:
            loss = cddpm.p_losses(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = batch["x0"].size(0)
            total_loss += loss.item() * bs

        avg = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:03d} | Loss: {avg:.6f}")


# =========================
# 8. 可视化：人工遮挡补全
# =========================
def plot_imputation_example(
    original: np.ndarray,
    masked: np.ndarray,
    imputed: np.ndarray,
    target_mask: np.ndarray,
    save_path: str = "cddpm_imputation_example.png"
):
    """
    输入 shape 都是 [C, L]
    """
    depth = np.arange(original.shape[1])

    fig, axes = plt.subplots(1, original.shape[0], figsize=(13, 6), sharey=True)

    for i, name in enumerate(LOG_NAMES):
        ax = axes[i]

        ax.plot(original[i], depth, label="Original")
        ax.plot(masked[i], depth, label="Condition")
        ax.plot(imputed[i], depth, linestyle="--", label="Imputed")

        masked_depth = np.where(target_mask[i] > 0.5)[0]
        if len(masked_depth) > 0:
            ax.axhspan(masked_depth.min(), masked_depth.max(), alpha=0.15)

        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Depth index")
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()
    print(f"Saved: {save_path}")


def inverse_one_window(x: np.ndarray, scaler: SimpleStandardScaler) -> np.ndarray:
    """
    x: [C, L] 标准化数据
    return: [C, L] 原始量纲
    """
    arr = x.T  # [L, C]
    inv = scaler.inverse_transform(arr)

    # RILD 从 log10 还原
    inv_df = pd.DataFrame(inv, columns=LOG_NAMES)
    inv_df["RILD"] = 10 ** inv_df["RILD"]

    return inv_df[LOG_NAMES].values.T


# =========================
# 9. 主流程
# =========================
def main():
    set_seed(SEED)

    print("Loading data...")
    raw_df = load_csv_or_las(DATA_PATH)

    logs_df = select_required_logs(raw_df)

    x_arr, obs_mask, scaler = preprocess_logs_keep_missing(logs_df)

    print("原始数据 shape:", x_arr.shape)
    print("观测比例:", obs_mask.mean())
    

    x_windows, m_windows = make_windows_with_mask(
        x_arr,
        obs_mask,
        WINDOW_SIZE,
        WINDOW_STRIDE
    )

    print("窗口数据 shape:", x_windows.shape)
    print("窗口 mask shape:", m_windows.shape)
    # return

    dataset = WellLogCDDPMDataset(
        x_windows,
        m_windows,
        target_min_len=TARGET_MIN_LEN,
        target_max_len=TARGET_MAX_LEN,
        min_target_channels=MIN_TARGET_CHANNELS,
        max_target_channels=MAX_TARGET_CHANNELS
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    model = ConditionalDenoiser1D(
        log_channels=4,
        hidden_channels=64,
        time_dim=128
    )

    cddpm = CDDPM(
        model=model,
        timesteps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=DEVICE
    )

    print(f"Training on {DEVICE}...")
    train(cddpm, dataloader, EPOCHS, LR)

    torch.save(cddpm.model.state_dict(), MODEL_DIR / "cddpm_welllog_model.pt")
    print("Saved model: cddpm_welllog_model.pt")

    # 做一个人工遮挡补全示例
    print("Running one imputation example...")
    sample = dataset[0]

    x0 = sample["x0"].unsqueeze(0).to(DEVICE)
    m_cond = sample["m_cond"].unsqueeze(0).to(DEVICE)
    m_ta = sample["m_ta"].unsqueeze(0).to(DEVICE)
    x_cond = sample["x_cond"].unsqueeze(0).to(DEVICE)

    imputed = cddpm.impute(
        x0_filled=x_cond,
        m_obs=m_cond,
        target_mask=m_ta
    )

    original_np = x0[0].detach().cpu().numpy()
    masked_np = x_cond[0].detach().cpu().numpy()
    imputed_np = imputed[0].detach().cpu().numpy()
    target_np = m_ta[0].detach().cpu().numpy()

    # 还原到原始量纲后画图
    original_inv = inverse_one_window(original_np, scaler)
    masked_inv = inverse_one_window(masked_np, scaler)
    imputed_inv = inverse_one_window(imputed_np, scaler)

    plot_imputation_example(
        original=original_inv,
        masked=masked_inv,
        imputed=imputed_inv,
        target_mask=target_np,
        save_path=IMAGE_DIR / "cddpm_imputation_example.png"
    )


if __name__ == "__main__":
    main()