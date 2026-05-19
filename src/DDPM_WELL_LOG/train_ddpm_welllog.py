import os
import math
from pathlib import Path
import random
from typing import List, Optional, Tuple, Dict, Any
import time
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


# ============================================================
# 0. 路径与环境设置
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGE_DIR = CURRENT_DIR / "images"
MODEL_DIR = CURRENT_DIR / "models"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LATEST_MODEL_PATH = MODEL_DIR / "latest_ddpm_welllog_unconditional.pth"
RUN_TS = int(time.time() * 1000)
# ============================================================
# 1. 配置
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = str(DATA_DIR / "well_log_data" / "log.csv")

LOG_COLUMNS_CANDIDATES = {
    "GR": ["GR", "GAMMA", "GAMMA_RAY"],
    "RHOB": ["RHOB", "RHOZ", "RHO_B"],
    "NPHI": ["NPHI", "NPHI_LS", "NPHI_N", "CNLS", "CNPOR", "NPLS"],
    "RILD": ["RILD", "RLLD", "ILD", "RT"],
}
LOG_NAMES = ["GR", "RHOB", "NPHI", "RILD"]

WINDOW_SIZE = 128
WINDOW_STRIDE = 32
BATCH_SIZE = 64
EPOCHS = 20              # 每次运行继续训练多少轮
LR = 1e-3

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02
NUM_WORKERS = 0
SAMPLE_SEED = 2026       # 固定这个种子，生成结果可复现


# ============================================================
# 2. 工具函数
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        return lasio.read(path).df().reset_index()
    raise ValueError("仅支持 .csv 或 .las 文件")


def select_required_logs(df: pd.DataFrame) -> pd.DataFrame:
    selected = {}
    matched_map = {}
    for std_name, candidates in LOG_COLUMNS_CANDIDATES.items():
        matched = find_matching_column(df, candidates)
        if matched is not None:
            selected[std_name] = pd.to_numeric(df[matched], errors="coerce")
            matched_map[std_name] = matched
    print("实际匹配到的列：", matched_map)

    missing = [k for k in LOG_NAMES if k not in selected]
    if missing:
        raise ValueError(f"缺少必要曲线列: {missing}，当前可识别列: {list(df.columns)}")

    out = pd.DataFrame(selected)
    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))

    if out["NPHI"].dropna().shape[0] > 0 and out["NPHI"].dropna().median() > 1.5:
        out["NPHI"] = out["NPHI"] / 100.0
    out["NPHI"] = out["NPHI"].mask((out["NPHI"] < -0.15) | (out["NPHI"] > 1.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)
    return out


def preprocess_logs(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    # 普通 DDPM 是无条件生成模型，需要完整训练样本，所以这里插值补齐缺失。
    df = df.copy()
    df["RILD"] = np.log10(df["RILD"])
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.dropna().reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"有效样本太少，清洗后仅 {len(df)} 行，小于窗口长度 {WINDOW_SIZE}")

    scaler = StandardScaler()
    arr = scaler.fit_transform(df[LOG_NAMES].values.astype(np.float32)).astype(np.float32)
    return arr, scaler


def make_windows(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(arr) - window_size + 1, stride):
        windows.append(arr[start:start + window_size])
    if not windows:
        raise RuntimeError("没有生成任何窗口，请检查 WINDOW_SIZE/WINDOW_STRIDE 和数据长度。")
    windows = np.stack(windows, axis=0)          # [N, L, C]
    windows = np.transpose(windows, (0, 2, 1))  # [N, C, L]
    return windows.astype(np.float32)


def scaler_to_state(scaler: StandardScaler) -> Dict[str, np.ndarray]:
    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "var": scaler.var_.astype(np.float32),
        "n_features_in": np.array([scaler.n_features_in_], dtype=np.int64),
    }


def scaler_from_state(state: Dict[str, np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = state["mean"]
    scaler.scale_ = state["scale"]
    scaler.var_ = state["var"]
    scaler.n_features_in_ = int(state["n_features_in"][0])
    return scaler


def current_timestamp() -> str:
    """返回毫秒级 LONG 时间戳字符串，用于图片命名，避免覆盖旧图片。"""
    return str(int(time.time() * 1000))


# ============================================================
# 3. Dataset
# ============================================================
class WellLogDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


# ============================================================
# 4. 网络模块
# ============================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.block1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.block2(h)
        return x + h


class SimpleDenoiser1D(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock1D(hidden_channels, time_dim)
        self.res2 = ResidualBlock1D(hidden_channels, time_dim)
        self.res3 = ResidualBlock1D(hidden_channels, time_dim)
        self.res4 = ResidualBlock1D(hidden_channels, time_dim)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(x)
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        h = self.res3(h, t_emb)
        h = self.res4(h, t_emb)
        return self.output_proj(h)


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise, noise

    def p_losses(self, x0: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        b = x.shape[0]
        t_batch = torch.full((b,), t, device=x.device, dtype=torch.long)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        pred_noise = self.model(x, t_batch)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
        if t > 0:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            return mean + torch.sqrt(beta_t) * noise
        return mean

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], sample_seed: Optional[int] = None) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(sample_seed)
        x = torch.randn(shape, device=device, generator=generator)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, generator=generator)
        return x


# ============================================================
# 5. 训练、加载、采样、绘图
# ============================================================
def build_data() -> Tuple[DataLoader, StandardScaler, np.ndarray]:
    print("Loading data...")
    raw_df = load_csv_or_las(DATA_PATH)
    logs_df = select_required_logs(raw_df)
    arr, scaler = preprocess_logs(logs_df)
    print(f"Cleaned data shape: {arr.shape}")
    windows = make_windows(arr, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"Windows shape: {windows.shape}")
    dataset = WellLogDataset(windows)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    return loader, scaler, windows


def build_model() -> DDPM:
    denoiser = SimpleDenoiser1D(in_channels=len(LOG_NAMES), hidden_channels=64, time_dim=128)
    return DDPM(denoiser, timesteps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END).to(DEVICE)


def save_checkpoint(ddpm: DDPM, optimizer: torch.optim.Optimizer, scaler: StandardScaler,
                    epoch: int, losses: List[float], train_times: int, total_training_time: float) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": ddpm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state": scaler_to_state(scaler),
        "losses": losses,
        "train_times": train_times,
        "total_training_time": total_training_time,
        "config": {
            "log_names": LOG_NAMES,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "diffusion_steps": DIFFUSION_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
        },
    }, LATEST_MODEL_PATH)


def train_model() -> None:
    import time
    set_seed(SEED)
    loader, scaler, _ = build_data()
    ddpm = build_model()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=LR)

    start_epoch = 0
    end_epoch = EPOCHS
    losses = []
    train_times = 0
    total_training_time = 0.0

    if LATEST_MODEL_PATH.exists():
        ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        try:
            ddpm.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            end_epoch = start_epoch + EPOCHS
            losses = ckpt.get("losses", [])
            train_times = int(ckpt.get("train_times", 0))
            total_training_time = float(ckpt.get("total_training_time", 0.0))
            if "scaler_state" in ckpt:
                scaler = scaler_from_state(ckpt["scaler_state"])
            print(f"已加载断点模型：start_epoch={start_epoch}, end_epoch={end_epoch}")
        except RuntimeError as e:
            print("旧 checkpoint 与当前模型结构不兼容，将从头训练。")
            print(str(e)[:1000])

    t0 = time.time()
    for epoch in range(start_epoch, end_epoch):
        ddpm.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.p_losses(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            bs = batch.size(0)
            total_n += bs
            total_loss += loss.item() * bs
        avg = total_loss / max(total_n, 1)
        losses.append(avg)
        print(f"Epoch [{epoch + 1}/{end_epoch}] | Loss: {avg:.6f}")
        save_checkpoint(ddpm, optimizer, scaler, epoch + 1, losses, train_times, total_training_time + time.time() - t0)

    elapsed = time.time() - t0
    total_training_time += elapsed
    save_checkpoint(ddpm, optimizer, scaler, end_epoch, losses, train_times + 1, total_training_time)
    print(f"训练完成，模型已保存: {LATEST_MODEL_PATH}")


def load_trained_model() -> Tuple[DDPM, StandardScaler, Dict[str, Any]]:
    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件: {LATEST_MODEL_PATH}")
    ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    ddpm = build_model()
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm.eval()
    scaler = scaler_from_state(ckpt["scaler_state"]) if "scaler_state" in ckpt else build_data()[1]
    return ddpm, scaler, ckpt


def inverse_transform_sample(sample: np.ndarray, scaler: StandardScaler) -> pd.DataFrame:
    arr = np.transpose(sample, (1, 0))
    arr = scaler.inverse_transform(arr)
    df = pd.DataFrame(arr, columns=["GR", "RHOB", "NPHI", "RILD_log10"])
    df["RILD"] = np.power(10.0, df["RILD_log10"])
    return df.drop(columns=["RILD_log10"])


def plot_generated_sample(df: pd.DataFrame, save_path: Optional[Path] = None, title: str = "") -> None:
    """
    绘制生成的测井曲线窗口。

    关键规则：
    1. 纵坐标表示深度点索引；
    2. 纵坐标数值从上到下增大，即测井曲线常规显示方式；
    3. 四条曲线使用不同颜色；
    4. 图片标题和文件名都带毫秒级 LONG 时间戳。
    """
    depth = np.arange(len(df), dtype=np.int32)

    curve_info = [
        ("GR", "green", "API"),
        ("RHOB", "red", "g/cm³"),
        ("NPHI", "blue", "fraction"),
        ("RILD", "purple", "ohm·m"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 7), sharey=True)

    for ax, (name, color, unit) in zip(axes, curve_info):
        ax.plot(df[name].values, depth, color=color, linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel(unit)
        ax.grid(True, alpha=0.3)

        # 测井曲线标准方向：上浅下深，纵坐标数值从上到下增大。
        ax.invert_yaxis()

    axes[0].set_ylabel("Depth index")

    if title:
        fig.suptitle(title, fontsize=13)
    else:
        fig.suptitle(f"Generated well-log window | ts={current_timestamp()}", fontsize=13)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"已保存图片: {save_path}")

    plt.show()


@torch.no_grad()
def sample_mode(sample_seed: int = SAMPLE_SEED, num_samples: int = 1) -> None:
    set_seed(SEED)
    ddpm, scaler, _ = load_trained_model()
    print(f"Sampling with fixed sample_seed={sample_seed} ...")
    samples = ddpm.sample((num_samples, len(LOG_NAMES), WINDOW_SIZE), sample_seed=sample_seed)
    ts = current_timestamp()
    for i in range(num_samples):
        df = inverse_transform_sample(samples[i].detach().cpu().numpy(), scaler)
        print(f"Sample {i + 1}:")
        print(df.head())

        save_path = IMAGE_DIR / f"generated_well_log_seed{sample_seed}_{ts}_{i + 1:02d}.png"
        title = f"Generated well-log window | seed={sample_seed} | sample={i + 1} | ts={ts}"
        plot_generated_sample(df, save_path=save_path, title=title)


def show_model_status() -> None:
    if not LATEST_MODEL_PATH.exists():
        print("尚未发现模型文件。")
        return
    ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    losses = ckpt.get("losses", [])
    print("\n" + "=" * 50)
    print("普通 DDPM 测井曲线生成模型状态")
    print(f"模型文件: {LATEST_MODEL_PATH}")
    print(f"已训练 epoch: {ckpt.get('epoch', 0)}")
    print(f"累计训练次数: {ckpt.get('train_times', 0)}")
    if losses:
        print(f"最近训练 loss: {losses[-1]:.6f}")
    print(f"累计训练耗时: {ckpt.get('total_training_time', 0.0) / 60:.2f} 分钟")
    print("=" * 50 + "\n")


def generate_loss_plot() -> None:
    if not LATEST_MODEL_PATH.exists():
        print("尚未发现模型文件，请先训练。")
        return
    ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    losses = ckpt.get("losses", [])
    if not losses:
        print("checkpoint 中没有 losses。")
        return
    epochs = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction MSE")
    plt.title(f"Unconditional DDPM Well-Log Training Loss | epochs={len(losses)}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path = IMAGE_DIR / f"ddpm_welllog_loss_{current_timestamp()}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Loss 图已保存: {save_path}")


# ============================================================
# 6. 主菜单
# ============================================================
def main() -> None:
    set_seed(SEED)
    while True:
        print("\n" + "=" * 50)
        print("普通 DDPM 测井曲线生成管理系统")
        print("[1] 训练模型 Train / Resume")
        print("[2] 采样生成 Sample")
        print("[3] 查看模型状态 Status")
        print("[4] 生成 Loss 曲线")
        print("[0/exit] 退出")
        print("=" * 50)
        choice = input("请选择功能: ").strip().lower()
        if choice == "1":
            train_model()
        elif choice == "2":
            seed_text = input(f"请输入采样随机种子，默认 {SAMPLE_SEED}: ").strip()
            sample_seed = int(seed_text) if seed_text else SAMPLE_SEED
            n_text = input("请输入生成样本数，默认 1: ").strip()
            num_samples = int(n_text) if n_text else 1
            sample_mode(sample_seed=sample_seed, num_samples=num_samples)
        elif choice == "3":
            show_model_status()
        elif choice == "4":
            generate_loss_plot()
        elif choice in ["0", "exit"]:
            print("退出程序。")
            break
        else:
            print("无效输入。")


if __name__ == "__main__":
    main()
