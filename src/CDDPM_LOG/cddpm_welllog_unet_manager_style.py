"""
简化版 WellLog CDDPM U-Net

功能：
1. 直接读取已有 well_file_split.npz，按 train/val/test 文件名构造数据集。
2. 训练阶段：只用 train wells 拟合 scaler，只用 train/val 训练和验证。
3. 测试阶段：只读取 test wells，人工挖缺失计算 MAE/RMSE/MSE，并保存测试图。
4. 案例生成：从 test wells 或手动指定井，输出真实缺失补全 CSV 和对比图。
5. 支持断点续训、模型状态查看、loss 图生成。

CSV 必须包含列：Depth, GR, RHOB, RILD, CNPOR
NPZ 必须包含数组：train, val, test，里面保存 csv 文件名，例如 xxx.csv
"""

import os
import sys
import math
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 避免某些 PyTorch 版本初始化 torch._dynamo 过慢
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

# ============================================================
# 0. 路径与日志：保留你的工程风格
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGE_DIR = CURRENT_DIR / "images"
MODEL_DIR = CURRENT_DIR / "models"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

try:
    from utils.logger import create_logger
    logger = create_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

try:
    from utils.gpu_info import init_gpu_environment
    DEVICE = init_gpu_environment()
except Exception:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. 配置区：主要改这里
# ============================================================
SEED = 42

CSV_DIR = DATA_DIR / "2023_log_csv"
SPLIT_NPZ_PATH = DATA_DIR / 'log_file_npz'  / "well_file_split.npz"

LOG_NAMES = ["GR", "RHOB", "RILD", "CNPOR"]
REQUIRED_COLUMNS = ["Depth"] + LOG_NAMES

WINDOW_SIZE = 128
WINDOW_STRIDE = 32
MIN_OBS_RATIO = 0.50

BATCH_SIZE = 64
NUM_EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

TARGET_MIN_LEN = 16
TARGET_MAX_LEN = 48
MIN_TARGET_CHANNELS = 1
MAX_TARGET_CHANNELS = 2

BASE_CHANNELS = 64
TIME_DIM = 128
DEPTH_CHANNELS = 16

EVAL_BATCHES = 4

LATEST_MODEL_PATH = MODEL_DIR / "latest_welllog_cddpm_unet.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_welllog_cddpm_unet.pth"
LOSS_PLOT_PATH = MODEL_DIR / "welllog_cddpm_unet_loss.png"
CASE_DIR = IMAGE_DIR / "welllog_case_figures"
CASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 基础工具
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


def find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    upper_map = {c.upper(): c for c in df.columns}
    return upper_map.get(name.upper())


def read_well_csv(csv_path: Path) -> pd.DataFrame:
    """读取一口井 CSV，并做基础清洗。"""
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()

    for col in REQUIRED_COLUMNS:
        matched = find_col(df, col)
        if matched is None:
            raise ValueError(f"缺少必要列: {col}")
        out[col] = pd.to_numeric(df[matched], errors="coerce")

    out = out.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)

    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)

    # CNPOR 有些 LAS 转出是百分数，例如 25，需要转成 0.25
    cnpor_valid = out["CNPOR"].dropna()
    if len(cnpor_valid) > 0 and cnpor_valid.median() > 1.5:
        out["CNPOR"] = out["CNPOR"] / 100.0
    out["CNPOR"] = out["CNPOR"].mask((out["CNPOR"] < -0.2) | (out["CNPOR"] > 1.0))

    # 电阻率 log10
    out["RILD"] = np.log10(out["RILD"])
    return out


def scan_csv_dir(csv_dir: Path) -> Dict[str, Path]:
    paths = sorted(list(Path(csv_dir).glob("*.csv")) + list(Path(csv_dir).glob("*.CSV")))
    if not paths:
        raise FileNotFoundError(f"CSV 文件夹为空: {csv_dir}")
    return {p.name: p for p in paths}


def load_split_files() -> Tuple[List[Path], List[Path], List[Path]]:
    """直接读取已有 npz 划分，不再在代码中重新划分。"""
    if not SPLIT_NPZ_PATH.exists():
        raise FileNotFoundError(f"找不到划分文件: {SPLIT_NPZ_PATH}")

    path_map = scan_csv_dir(Path(CSV_DIR))
    split = np.load(SPLIT_NPZ_PATH, allow_pickle=True)

    def to_paths(key: str) -> List[Path]:
        names = [str(x) for x in split[key]]
        paths = [path_map[n] for n in names if n in path_map]
        missing = [n for n in names if n not in path_map]
        if missing:
            logger.warning(f"{key} 中有 {len(missing)} 个文件在 CSV_DIR 中不存在，已跳过。")
        return paths

    train_files = to_paths("train")
    val_files = to_paths("val")
    test_files = to_paths("test")
    logger.info(f"读取 NPZ 划分完成: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    return train_files, val_files, test_files


def fit_scaler_from_train_files(train_files: List[Path]) -> Tuple[SimpleStandardScaler, float, float]:
    """只使用训练井拟合 scaler，避免验证/测试信息泄漏。"""
    all_logs, all_depth = [], []
    for p in train_files:
        try:
            df = read_well_csv(p)
            if len(df) < WINDOW_SIZE:
                continue
            values = df[LOG_NAMES].values.astype(np.float32)
            if np.isfinite(values).mean() < MIN_OBS_RATIO:
                continue
            all_logs.append(values)
            all_depth.append(df["Depth"].values.astype(np.float32).reshape(-1, 1))
        except Exception as e:
            logger.info(f"[SCALER-SKIP] {p.name}: {e}")

    if not all_logs:
        raise RuntimeError("训练井中没有可用于拟合 scaler 的有效数据。")

    logs = np.concatenate(all_logs, axis=0)
    depths = np.concatenate(all_depth, axis=0)
    log_scaler = SimpleStandardScaler().fit(logs)
    depth_mean = float(np.nanmean(depths))
    depth_std = float(np.nanstd(depths))
    if depth_std < 1e-6:
        depth_std = 1.0

    logger.info(f"Scaler 拟合完成: obs_ratio={np.isfinite(logs).mean():.4f}, depth_mean={depth_mean:.2f}, depth_std={depth_std:.2f}")
    return log_scaler, depth_mean, depth_std


def build_windows(
    files: List[Path],
    log_scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """按井内部滑窗，最后合并窗口样本。"""
    all_x, all_m, all_d, valid_files = [], [], [], []

    for p in files:
        try:
            df = read_well_csv(p)
            if len(df) < WINDOW_SIZE:
                logger.info(f"[SKIP-short-{split_name}] {p.name}: length={len(df)}")
                continue

            values = df[LOG_NAMES].values.astype(np.float32)
            obs_mask = np.isfinite(values).astype(np.float32)
            if obs_mask.mean() < MIN_OBS_RATIO:
                logger.info(f"[SKIP-lowobs-{split_name}] {p.name}: obs={obs_mask.mean():.3f}")
                continue

            x_scaled = log_scaler.transform(values).astype(np.float32)
            x_filled = np.where(np.isfinite(x_scaled), x_scaled, 0.0).astype(np.float32)
            depth = df["Depth"].values.astype(np.float32)

            x_list, m_list, d_list = [], [], []
            for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                end = start + WINDOW_SIZE
                xw = x_filled[start:end]
                mw = obs_mask[start:end]
                if mw.mean() < MIN_OBS_RATIO:
                    continue

                rel_depth = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
                abs_depth = ((depth[start:end] - depth_mean) / depth_std).astype(np.float32)
                dw = np.stack([rel_depth, abs_depth], axis=0)

                x_list.append(xw)
                m_list.append(mw)
                d_list.append(dw)

            if not x_list:
                logger.info(f"[NO-WINDOW-{split_name}] {p.name}")
                continue

            x_arr = np.transpose(np.stack(x_list), (0, 2, 1))  # [N,C,L]
            m_arr = np.transpose(np.stack(m_list), (0, 2, 1))  # [N,C,L]
            d_arr = np.stack(d_list)                           # [N,2,L]

            all_x.append(x_arr.astype(np.float32))
            all_m.append(m_arr.astype(np.float32))
            all_d.append(d_arr.astype(np.float32))
            valid_files.append(p)
            logger.info(f"[WINDOW-{split_name}] {p.name}: {len(x_arr)}")
        except Exception as e:
            logger.info(f"[SKIP-error-{split_name}] {p.name}: {e}")

    if not all_x:
        raise RuntimeError(f"{split_name} 没有生成任何窗口。")

    x = np.concatenate(all_x, axis=0)
    m = np.concatenate(all_m, axis=0)
    d = np.concatenate(all_d, axis=0)
    logger.info(f"{split_name} windows: x={x.shape}, mask={m.shape}, depth={d.shape}")
    return x, m, d, valid_files


# ============================================================
# 3. Dataset：动态人工挖缺失
# ============================================================
class WellLogDataset(Dataset):
    def __init__(self, x: np.ndarray, m_obs: np.ndarray, depth: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.m_obs = torch.from_numpy(m_obs).float()
        self.depth = torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.x)

    def _target_mask(self, m_obs: torch.Tensor) -> torch.Tensor:
        c, l = m_obs.shape
        m_ta = torch.zeros_like(m_obs)
        num_ch = random.randint(MIN_TARGET_CHANNELS, min(MAX_TARGET_CHANNELS, c))
        channels = random.sample(range(c), k=num_ch)

        for ch in channels:
            seg_len = random.randint(TARGET_MIN_LEN, TARGET_MAX_LEN)
            seg_len = min(seg_len, max(1, l // 2))

            # 人工挖缺失必须避开真实缺失，target 区域原本要有真实值
            ok = False
            for _ in range(50):
                start = random.randint(0, l - seg_len)
                end = start + seg_len
                if m_obs[ch, start:end].mean() > 0.95:
                    m_ta[ch, start:end] = 1.0
                    ok = True
                    break

            # 找不到连续完整段时，退化成从有值点里取一小段
            if not ok:
                valid = torch.where(m_obs[ch] > 0.5)[0]
                if len(valid) > seg_len:
                    start_pos = random.randint(0, len(valid) - seg_len)
                    chosen = valid[start_pos:start_pos + seg_len]
                    m_ta[ch, chosen] = 1.0

        return m_ta

    def __getitem__(self, idx):
        x0 = self.x[idx].clone()
        m_obs = self.m_obs[idx].clone()
        depth = self.depth[idx].clone()
        m_ta = self._target_mask(m_obs)
        m_cond = m_obs * (1.0 - m_ta)
        x_cond = x0 * m_cond
        return {"x0": x0, "m_obs": m_obs, "m_ta": m_ta, "m_cond": m_cond, "x_cond": x_cond, "depth": depth}


def make_loader(x: np.ndarray, m: np.ndarray, d: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        WellLogDataset(x, m, d),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )


# ============================================================
# 4. 1D Conditional U-Net
# ============================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, self.dim - emb.shape[-1])) if emb.shape[-1] < self.dim else emb


def gn(ch: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = gn(out_ch)
        self.norm2 = gn(out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Down1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block1 = ResBlock1D(in_ch, out_ch, time_dim)
        self.block2 = ResBlock1D(out_ch, out_ch, time_dim)
        self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return self.down(x), x


class Up1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResBlock1D(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = ResBlock1D(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        return self.block2(x, t_emb)


class ConditionalUNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(TIME_DIM), nn.Linear(TIME_DIM, TIME_DIM), nn.SiLU(), nn.Linear(TIME_DIM, TIME_DIM)
        )
        self.depth_proj = nn.Sequential(
            nn.Conv1d(2, DEPTH_CHANNELS, 1), nn.SiLU(),
            nn.Conv1d(DEPTH_CHANNELS, DEPTH_CHANNELS, 3, padding=1), nn.SiLU(),
        )

        in_ch = len(LOG_NAMES) * 3 + DEPTH_CHANNELS
        bc = BASE_CHANNELS
        self.init = nn.Conv1d(in_ch, bc, 3, padding=1)
        self.down1 = Down1D(bc, bc, TIME_DIM)
        self.down2 = Down1D(bc, bc * 2, TIME_DIM)
        self.down3 = Down1D(bc * 2, bc * 4, TIME_DIM)
        self.mid1 = ResBlock1D(bc * 4, bc * 4, TIME_DIM)
        self.mid2 = ResBlock1D(bc * 4, bc * 4, TIME_DIM)
        self.up3 = Up1D(bc * 4, bc * 4, bc * 2, TIME_DIM)
        self.up2 = Up1D(bc * 2, bc * 2, bc, TIME_DIM)
        self.up1 = Up1D(bc, bc, bc, TIME_DIM)
        self.out = nn.Sequential(gn(bc), nn.SiLU(), nn.Conv1d(bc, len(LOG_NAMES), 1))

    def forward(self, x_ta_t, x_cond, m_cond, depth, t):
        t_emb = self.time_emb(t)
        d_emb = self.depth_proj(depth)
        x = torch.cat([x_ta_t, x_cond, m_cond, d_emb], dim=1)
        x = self.init(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)
        x = self.mid2(self.mid1(x, t_emb), t_emb)
        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out(x)


# ============================================================
# 5. CDDPM
# ============================================================
class CDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ConditionalUNet1D()
        self.timesteps = DIFFUSION_STEPS

        betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise, noise

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = batch["x0"].to(DEVICE)
        x_cond = batch["x_cond"].to(DEVICE)
        m_cond = batch["m_cond"].to(DEVICE)
        m_ta = batch["m_ta"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)

        t = torch.randint(0, self.timesteps, (x0.size(0),), device=DEVICE).long()
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.model(xt * m_ta, x_cond, m_cond, depth, t)
        return (((pred_noise - noise) ** 2) * m_ta).sum() / m_ta.sum().clamp(min=1.0)

    @torch.no_grad()
    def impute(self, x0_filled: torch.Tensor, m_obs: torch.Tensor, depth: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        self.eval()
        x0_filled = x0_filled.to(DEVICE)
        m_obs = m_obs.to(DEVICE)
        depth = depth.to(DEVICE)

        m_target = (1.0 - m_obs) if target_mask is None else target_mask.to(DEVICE)
        m_cond = 1.0 - m_target
        x_cond = x0_filled * m_cond
        x = torch.randn_like(x0_filled) * m_target

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=DEVICE, dtype=torch.long)
            pred_noise = self.model(x, x_cond, m_cond, depth, t)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t))
            x = mean if step == 0 else mean + torch.sqrt(beta_t) * torch.randn_like(x)
            x = x * m_target

        return x_cond + x * m_target


# ============================================================
# 6. 构造 loader / checkpoint
# ============================================================
def build_loaders(mode: str, checkpoint: Optional[Dict] = None):
    train_files, val_files, test_files = load_split_files()

    if checkpoint is None:
        log_scaler, depth_mean, depth_std = fit_scaler_from_train_files(train_files)
    else:
        log_scaler = SimpleStandardScaler()
        log_scaler.load_state_dict(checkpoint["log_scaler"])
        depth_mean = float(checkpoint["depth_mean"])
        depth_std = float(checkpoint["depth_std"])

    if mode == "train":
        train_x, train_m, train_d, _ = build_windows(train_files, log_scaler, depth_mean, depth_std, "train")
        val_x, val_m, val_d, _ = build_windows(val_files, log_scaler, depth_mean, depth_std, "val")
        return make_loader(train_x, train_m, train_d, True), make_loader(val_x, val_m, val_d, False), log_scaler, depth_mean, depth_std

    if mode == "test":
        test_x, test_m, test_d, _ = build_windows(test_files, log_scaler, depth_mean, depth_std, "test")
        return make_loader(test_x, test_m, test_d, False), test_files, log_scaler, depth_mean, depth_std

    raise ValueError("mode must be train or test")


def save_checkpoint(path: Path, model: CDDPM, optimizer, epoch: int, train_losses: List[float], val_losses: List[float],
                    best_val: float, train_times: int, total_time: float,
                    log_scaler: SimpleStandardScaler, depth_mean: float, depth_std: float):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "train_times": train_times,
        "total_training_time": total_time,
        "log_scaler": log_scaler.state_dict(),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "log_names": LOG_NAMES,
        "window_size": WINDOW_SIZE,
        "diffusion_steps": DIFFUSION_STEPS,
    }, path)


def load_checkpoint(prefer_best: bool = True):
    path = BEST_MODEL_PATH if prefer_best and BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("未找到模型，请先训练。")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CDDPM().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"已加载模型: {path.name}")
    return model, ckpt


@torch.no_grad()
def estimate_loss(model: CDDPM, loader: DataLoader, max_batches: int = 10) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(float(model(batch).item()))
    return float(np.mean(losses)) if losses else float("nan")


# ============================================================
# 7. 训练 / 测试 / 生成
# ============================================================
def train_model():
    set_seed(SEED)
    start_epoch, train_times, total_time = 0, 0, 0.0
    train_losses, val_losses = [], []
    best_val = float("inf")

    model = CDDPM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint = None

    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        train_times = int(checkpoint.get("train_times", 0))
        total_time = float(checkpoint.get("total_training_time", 0.0))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        logger.info(f"断点续训：start_epoch={start_epoch}, 本次训练到 {start_epoch + NUM_EPOCHS}")
    else:
        logger.info("第一次训练模型。")

    train_loader, val_loader, log_scaler, depth_mean, depth_std = build_loaders("train", checkpoint)
    tic = time.time()
    end_epoch = start_epoch + NUM_EPOCHS

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_n = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = batch["x0"].size(0)
            total_loss += loss.item() * bs
            total_n += bs

        train_loss = total_loss / max(total_n, 1)
        val_loss = estimate_loss(model, val_loader, max_batches=10)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - tic

        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        save_checkpoint(
            LATEST_MODEL_PATH, model, optimizer, epoch + 1, train_losses, val_losses, best_val,
            train_times, total_time + elapsed, log_scaler, depth_mean, depth_std,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                BEST_MODEL_PATH, model, optimizer, epoch + 1, train_losses, val_losses, best_val,
                train_times, total_time + elapsed, log_scaler, depth_mean, depth_std,
            )
            logger.info(f"验证集 loss 降低，保存 best: {BEST_MODEL_PATH.name}")

    total_time += time.time() - tic
    save_checkpoint(
        LATEST_MODEL_PATH, model, optimizer, end_epoch, train_losses, val_losses, best_val,
        train_times + 1, total_time, log_scaler, depth_mean, depth_std,
    )
    logger.info(f"训练完成，本次耗时 {(time.time() - tic):.2f}s，累计训练 {total_time / 60:.2f} 分钟。")


@torch.no_grad()
def evaluate_imputation(model: CDDPM, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    se_sum, ae_sum, n_sum = 0.0, 0.0, 0.0

    for i, batch in enumerate(loader):
        if i >= EVAL_BATCHES:
            break
        x0 = batch["x0"].to(DEVICE)
        m_obs = batch["m_obs"].to(DEVICE)
        m_ta = batch["m_ta"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)

        x_masked = x0 * (1.0 - m_ta)
        m_cond = m_obs * (1.0 - m_ta)
        x_hat = model.impute(x_masked, m_cond, depth, target_mask=m_ta)
        diff = (x_hat - x0) * m_ta

        se_sum += float((diff ** 2).sum().item())
        ae_sum += float(diff.abs().sum().item())
        n_sum += float(m_ta.sum().item())

    mse = se_sum / max(n_sum, 1.0)
    return {"MAE_std": ae_sum / max(n_sum, 1.0), "RMSE_std": math.sqrt(mse), "MSE_std": mse}


@torch.no_grad()
def test_model():
    model, ckpt = load_checkpoint(prefer_best=True)
    test_loader, _, _, _, _ = build_loaders("test", ckpt)
    metrics = evaluate_imputation(model, test_loader)

    logger.info("\n" + "=" * 50)
    logger.info(f"{'测试集人工缺失补全指标':^40}")
    for k, v in metrics.items():
        logger.info(f" {k:<10}: {v:.6f}")
    logger.info("=" * 50)

    save_path = IMAGE_DIR / f"welllog_test_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_artificial_batch(model, test_loader, ckpt, save_path)


@torch.no_grad()
def impute_one_well(model: CDDPM, csv_path: Path, ckpt: Dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    log_scaler = SimpleStandardScaler()
    log_scaler.load_state_dict(ckpt["log_scaler"])
    depth_mean = float(ckpt["depth_mean"])
    depth_std = float(ckpt["depth_std"])

    df = read_well_csv(csv_path)
    values = df[LOG_NAMES].values.astype(np.float32)
    m = np.isfinite(values).astype(np.float32)
    x_scaled = log_scaler.transform(values).astype(np.float32)
    x = np.where(np.isfinite(x_scaled), x_scaled, 0.0).astype(np.float32)
    depth = df["Depth"].values.astype(np.float32)

    pred_sum = np.zeros_like(x, dtype=np.float32)
    pred_count = np.zeros_like(x, dtype=np.float32)
    starts = list(range(0, len(df) - WINDOW_SIZE + 1, WINDOW_STRIDE)) if len(df) >= WINDOW_SIZE else []
    if len(df) >= WINDOW_SIZE and (not starts or starts[-1] != len(df) - WINDOW_SIZE):
        starts.append(len(df) - WINDOW_SIZE)

    for start in starts:
        end = start + WINDOW_SIZE
        rel_depth = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
        abs_depth = ((depth[start:end] - depth_mean) / depth_std).astype(np.float32)
        d = np.stack([rel_depth, abs_depth], axis=0)

        x_t = torch.from_numpy(x[start:end].T[None]).float().to(DEVICE)
        m_t = torch.from_numpy(m[start:end].T[None]).float().to(DEVICE)
        d_t = torch.from_numpy(d[None]).float().to(DEVICE)

        completed = model.impute(x_t, m_t, d_t)[0].cpu().numpy().T
        missing = 1.0 - m[start:end]
        pred_sum[start:end] += completed * missing
        pred_count[start:end] += missing

    x_completed = x.copy()
    fill = pred_count > 0
    x_completed[fill] = pred_sum[fill] / np.maximum(pred_count[fill], 1e-6)
    logs_completed = log_scaler.inverse_transform(x_completed)

    out = df.copy()
    for j, name in enumerate(LOG_NAMES):
        raw = df[name].values.astype(np.float32)
        missing = ~np.isfinite(raw)
        out[f"{name}_original"] = raw
        out[f"{name}_imputed"] = logs_completed[:, j]
        out[name] = np.where(missing, logs_completed[:, j], raw)

    return out, depth, m, logs_completed


# ============================================================
# 8. 画图
# ============================================================
def inverse_window(arr_std: np.ndarray, ckpt: Dict) -> np.ndarray:
    scaler = SimpleStandardScaler()
    scaler.load_state_dict(ckpt["log_scaler"])
    return scaler.inverse_transform(arr_std.T).T


@torch.no_grad()
def plot_artificial_batch(model: CDDPM, loader: DataLoader, ckpt: Dict, save_path: Path):
    batch = next(iter(loader))
    x0 = batch["x0"][:1].to(DEVICE)
    m_obs = batch["m_obs"][:1].to(DEVICE)
    m_ta = batch["m_ta"][:1].to(DEVICE)
    depth = batch["depth"][:1].to(DEVICE)

    x_masked = x0 * (1.0 - m_ta)
    m_cond = m_obs * (1.0 - m_ta)
    x_hat = model.impute(x_masked, m_cond, depth, target_mask=m_ta)

    original = inverse_window(x0[0].cpu().numpy(), ckpt)
    masked = inverse_window(x_masked[0].cpu().numpy(), ckpt)
    imputed = inverse_window(x_hat[0].cpu().numpy(), ckpt)
    target = m_ta[0].cpu().numpy()
    depth_abs = depth[0, 1].cpu().numpy() * float(ckpt["depth_std"]) + float(ckpt["depth_mean"])

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 7), sharey=True)
    for i, name in enumerate(LOG_NAMES):
        ax = axes[i]
        ax.plot(original[i], depth_abs, label="original")
        ax.plot(masked[i], depth_abs, label="condition")
        ax.plot(imputed[i], depth_abs, linestyle="--", label="imputed")
        idx = np.where(target[i] > 0.5)[0]
        if len(idx) > 0:
            ax.axhspan(depth_abs[idx.min()], depth_abs[idx.max()], alpha=0.15)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"测试对比图已保存: {save_path}")


def plot_real_well(csv_path: Path, depth: np.ndarray, m: np.ndarray, completed: np.ndarray, save_path: Path):
    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 8), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        observed = completed[:, j].copy()
        observed[m[:, j] < 0.5] = np.nan
        ax.plot(observed, depth, label="observed")
        ax.plot(completed[:, j], depth, linestyle="--", label="observed + imputed")

        miss_idx = np.where(m[:, j] < 0.5)[0]
        if len(miss_idx) > 0:
            for seg in np.split(miss_idx, np.where(np.diff(miss_idx) != 1)[0] + 1):
                ax.axhspan(depth[seg.min()], depth[seg.max()], alpha=0.12)

        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[-1].legend()
    fig.suptitle(f"Real missing imputation: {csv_path.name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"真实缺失补全图已保存: {save_path}")


@torch.no_grad()
def generate_cases():
    model, ckpt = load_checkpoint(prefer_best=True)
    _, _, test_files = load_split_files()

    raw_num = input("请输入案例井数量 1~3，默认 3: ").strip()
    num_cases = min(max(int(raw_num) if raw_num.isdigit() else 3, 1), 3)
    raw_names = input("可选：指定井名，逗号分隔；直接回车则使用测试集前几口井: ").strip()

    if raw_names:
        name_map = scan_csv_dir(Path(CSV_DIR))
        stem_map = {p.stem: p for p in name_map.values()}
        selected = []
        for item in raw_names.split(","):
            key = item.strip()
            p = name_map.get(key) or stem_map.get(key)
            if p is not None:
                selected.append(p)
            else:
                logger.warning(f"找不到指定井: {key}")
        selected = selected[:num_cases]
    else:
        selected = test_files[:num_cases]

    for p in selected:
        out, depth, m, completed = impute_one_well(model, p, ckpt)
        out_csv = MODEL_DIR / f"{p.stem}_imputed.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        plot_real_well(p, depth, m, completed, CASE_DIR / f"{p.stem}_real_missing_imputed.png")
        logger.info(f"补全 CSV 已保存: {out_csv}")


# ============================================================
# 9. 状态 / loss / 菜单
# ============================================================
def show_status():
    path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    if not path.exists():
        logger.info("尚未发现模型文件。")
        return
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    logger.info("\n" + "=" * 55)
    logger.info(f"{'WellLog CDDPM U-Net 状态':^45}")
    logger.info(f" 模型文件       : {path.name}")
    logger.info(f" 已训练总轮数   : {ckpt.get('epoch', 0)}")
    logger.info(f" 累计训练次数   : {ckpt.get('train_times', 0)}")
    if train_losses:
        logger.info(f" 最近训练 Loss  : {train_losses[-1]:.6f}")
    if val_losses:
        logger.info(f" 最近验证 Loss  : {val_losses[-1]:.6f}")
    logger.info(f" 最优验证 Loss  : {ckpt.get('best_val_loss', float('nan')):.6f}")
    logger.info(f" 训练总时长     : {ckpt.get('total_training_time', 0.0) / 60:.2f} 分钟")
    logger.info(f" CSV_DIR        : {CSV_DIR}")
    logger.info(f" SPLIT_NPZ      : {SPLIT_NPZ_PATH}")
    logger.info("=" * 55)


def plot_loss():
    path = LATEST_MODEL_PATH if LATEST_MODEL_PATH.exists() else BEST_MODEL_PATH
    if not path.exists():
        logger.info("尚未发现模型文件，请先训练。")
        return
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    if not train_losses:
        logger.info("checkpoint 中没有 loss 记录。")
        return

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Noise Loss")
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, label="Val Noise Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction MSE")
    plt.title("WellLog CDDPM U-Net Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()
    logger.info(f"Loss 图已保存: {LOSS_PLOT_PATH}")


def set_csv_dir():
    global CSV_DIR
    raw = input(f"请输入 CSV 文件夹路径，当前为 {CSV_DIR}: ").strip()
    if raw:
        CSV_DIR = Path(raw)
        logger.info(f"CSV_DIR 已修改为: {CSV_DIR}")


def main():
    set_seed(SEED)
    logger.info(f"Device: {DEVICE}")

    while True:
        logger.info("\n" + "=" * 50)
        logger.info("        WellLog CDDPM U-Net 简化管理系统")
        logger.info(" [1] 训练模型 Train（断点续训）")
        logger.info(" [2] 测试指标 Test Metrics")
        logger.info(" [3] 生成案例 Generate Cases")
        logger.info(" [4] 查看模型状态 Status")
        logger.info(" [5] 生成 Loss 图")
        logger.info(" [6] 修改 CSV_DIR")
        logger.info(" [0/exit] 退出")
        logger.info("=" * 50)

        choice = input("请选择功能: ").strip().lower()
        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            generate_cases()
        elif choice == "4":
            show_status()
        elif choice == "5":
            plot_loss()
        elif choice == "6":
            set_csv_dir()
        elif choice in ["0", "exit"]:
            logger.info("退出程序。")
            break
        else:
            logger.info("无效输入。")


if __name__ == "__main__":
    main()
