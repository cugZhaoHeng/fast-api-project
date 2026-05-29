"""
CDDPM Well-log Imputation with 1D Conditional U-Net

Required CSV columns:
    Depth, GR, RHOB, RILD, CNPOR

Main points:
    1. Use existing npz split file: train/test or train/val/test.
       If val exists, val files are merged into train.
    2. Train with artificial masking only on originally observed values.
    3. Test by artificial masking on held-out test wells and compute MAE/RMSE/MSE.
    4. Generate case figures:
       - whole-well overview with real Depth axis and physical units
       - zoomed real-missing interval with real Depth axis and physical units
       - local multi-sample uncertainty plot, not whole-well repeated sampling
    5. Important plotting rule:
       - observed curve always comes from raw CSV physical values
       - imputed curve only replaces original missing points
       - Depth axis always uses df["Depth"], never index or relative depth for real-well plots

Author: Jack Zhao style refactor
"""

import os
import sys
import math
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Avoid slow torch dynamo init on some PyTorch versions.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Paths / logger / device
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
# 1. Config
# ============================================================
SEED = 42

CSV_DIR = DATA_DIR / "2023_log_csv"
SPLIT_NPZ_PATH = MODEL_DIR / "well_file_split_8_1_1.npz"

LOG_NAMES = ["GR", "RHOB", "RILD", "CNPOR"]
REQUIRED_COLUMNS = ["Depth"] + LOG_NAMES

# Physical units for plotting.
LOG_UNITS = {
    "Depth": "m",
    "GR": "API",
    "RHOB": "g/cm³",
    "RILD": "ohm·m",
    "CNPOR": "fraction",
}

WINDOW_SIZE = 128
WINDOW_STRIDE = 32
MIN_OBS_RATIO = 0.50

BATCH_SIZE = 64
NUM_EPOCHS = 100
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

EVAL_BATCHES = 6
OVERVIEW_POINTS = 400      # whole-well overview downsample points
ZOOM_CONTEXT_POINTS = 80   # points before/after missing interval
LOCAL_MC_SAMPLES = 5       # multi-sample local uncertainty

LATEST_MODEL_PATH = MODEL_DIR / "latest_welllog_cddpm_unet.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_welllog_cddpm_unet.pth"
LOSS_PLOT_PATH = MODEL_DIR / "welllog_cddpm_unet_loss.png"
CASE_DIR = IMAGE_DIR / "welllog_case_figures"
CASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Utils / scaler / CSV reading
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleStandardScaler:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = state["mean"]
        self.std = state["std"]


def find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    col_map = {c.upper(): c for c in df.columns}
    return col_map.get(target.upper())


def scan_csv_files(csv_dir: Path) -> List[Path]:
    paths = sorted(list(Path(csv_dir).glob("*.csv")) + list(Path(csv_dir).glob("*.CSV")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    return paths


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    """Read one well CSV and clean invalid values.

    Notes:
        - Raw physical units are preserved in returned df, except invalid values are converted to NaN.
        - RILD is still in physical ohm·m here. Log10 transform is only used internally for model input.
    """
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    for col in REQUIRED_COLUMNS:
        matched = find_col(df, col)
        if matched is None:
            raise ValueError(f"missing required column: {col}")
        out[col] = pd.to_numeric(df[matched], errors="coerce")

    out = out.dropna(subset=["Depth"])
    out = out.sort_values("Depth").reset_index(drop=True)

    # Basic physical cleaning.
    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)

    # CNPOR sometimes is percent, e.g. 25 -> 0.25.
    cnp = out["CNPOR"].dropna()
    if len(cnp) > 0 and cnp.median() > 1.5:
        out["CNPOR"] = out["CNPOR"] / 100.0
    out["CNPOR"] = out["CNPOR"].mask((out["CNPOR"] < -0.2) | (out["CNPOR"] > 1.0))
    return out


def physical_to_model_values(df: pd.DataFrame) -> np.ndarray:
    """Convert physical logs to model space. Only RILD is log10-transformed."""
    values = df[LOG_NAMES].values.astype(np.float32).copy()
    rild_idx = LOG_NAMES.index("RILD")
    values[:, rild_idx] = np.where(np.isfinite(values[:, rild_idx]), np.log10(values[:, rild_idx]), np.nan)
    return values


def model_to_physical_values(values_model: np.ndarray, scaler: SimpleStandardScaler) -> np.ndarray:
    """Inverse scaler and restore RILD from log10 to physical ohm·m."""
    values = scaler.inverse_transform(values_model)
    rild_idx = LOG_NAMES.index("RILD")
    values[:, rild_idx] = np.where(np.isfinite(values[:, rild_idx]), np.power(10.0, values[:, rild_idx]), np.nan)
    return values


def load_split_files() -> Tuple[List[Path], List[Path]]:
    """Load existing npz split. If val exists, merge val into train."""
    if not SPLIT_NPZ_PATH.exists():
        raise FileNotFoundError(f"Split npz not found: {SPLIT_NPZ_PATH}")

    all_files = scan_csv_files(CSV_DIR)
    name_map = {p.name: p for p in all_files}
    data = np.load(SPLIT_NPZ_PATH, allow_pickle=True)

    train_names = [str(x) for x in data["train"]] if "train" in data else []
    val_names = [str(x) for x in data["val"]] if "val" in data else []
    test_names = [str(x) for x in data["test"]] if "test" in data else []

    train_files = [name_map[n] for n in train_names + val_names if n in name_map]
    test_files = [name_map[n] for n in test_names if n in name_map]

    if not train_files or not test_files:
        raise RuntimeError(f"Invalid split file. train={len(train_files)}, test={len(test_files)}")

    logger.info(f"Loaded split npz: train={len(train_files)}, test={len(test_files)}")
    return train_files, test_files


def fit_scalers(train_files: List[Path]) -> Tuple[SimpleStandardScaler, float, float]:
    all_logs, all_depth = [], []
    for p in train_files:
        try:
            df = read_one_csv(p)
            if len(df) < WINDOW_SIZE:
                continue
            vals = physical_to_model_values(df)
            if np.isfinite(vals).mean() < MIN_OBS_RATIO:
                continue
            all_logs.append(vals)
            all_depth.append(df["Depth"].values.astype(np.float32).reshape(-1, 1))
        except Exception as e:
            logger.info(f"[SCALER-SKIP] {p.name}: {e}")

    if not all_logs:
        raise RuntimeError("No valid train wells for scaler fitting.")

    log_scaler = SimpleStandardScaler().fit(np.concatenate(all_logs, axis=0))
    depth_all = np.concatenate(all_depth, axis=0)
    depth_mean = float(np.nanmean(depth_all))
    depth_std = float(np.nanstd(depth_all))
    if depth_std < 1e-6:
        depth_std = 1.0

    logger.info(f"Scaler fitted on train wells. depth_mean={depth_mean:.3f}, depth_std={depth_std:.3f}")
    return log_scaler, depth_mean, depth_std


def preprocess_well_for_model(df: pd.DataFrame, scaler: SimpleStandardScaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals_model = physical_to_model_values(df)
    mask = np.isfinite(vals_model).astype(np.float32)
    vals_scaled = scaler.transform(vals_model).astype(np.float32)
    vals_filled = np.where(np.isfinite(vals_scaled), vals_scaled, 0.0).astype(np.float32)
    depth = df["Depth"].values.astype(np.float32)
    return vals_filled, mask, depth


def make_windows(
    x: np.ndarray,
    m: np.ndarray,
    depth: np.ndarray,
    depth_mean: float,
    depth_std: float,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    min_obs_ratio: float = MIN_OBS_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    xs, ms, ds, spans = [], [], [], []
    n = len(x)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        xw = x[start:end]
        mw = m[start:end]
        dw_abs = depth[start:end]
        if mw.mean() < min_obs_ratio:
            continue

        rel_depth = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
        abs_depth_scaled = ((dw_abs - depth_mean) / depth_std).astype(np.float32)
        dw = np.stack([rel_depth, abs_depth_scaled], axis=0).astype(np.float32)
        xs.append(xw)
        ms.append(mw)
        ds.append(dw)
        spans.append((start, end))

    if not xs:
        return (
            np.empty((0, len(LOG_NAMES), window_size), dtype=np.float32),
            np.empty((0, len(LOG_NAMES), window_size), dtype=np.float32),
            np.empty((0, 2, window_size), dtype=np.float32),
            [],
        )

    xs = np.transpose(np.stack(xs, axis=0), (0, 2, 1)).astype(np.float32)
    ms = np.transpose(np.stack(ms, axis=0), (0, 2, 1)).astype(np.float32)
    ds = np.stack(ds, axis=0).astype(np.float32)
    return xs, ms, ds, spans


def build_loader(files: List[Path], scaler: SimpleStandardScaler, depth_mean: float, depth_std: float, shuffle: bool) -> DataLoader:
    all_x, all_m, all_d = [], [], []
    for p in files:
        try:
            df = read_one_csv(p)
            if len(df) < WINDOW_SIZE:
                continue
            x, m, depth = preprocess_well_for_model(df, scaler)
            xw, mw, dw, _ = make_windows(x, m, depth, depth_mean, depth_std)
            if len(xw) > 0:
                all_x.append(xw)
                all_m.append(mw)
                all_d.append(dw)
            logger.info(f"[WINDOW] {p.name}: {len(xw)}")
        except Exception as e:
            logger.info(f"[WINDOW-SKIP] {p.name}: {e}")

    if not all_x:
        raise RuntimeError("No windows generated.")

    x = np.concatenate(all_x, axis=0)
    m = np.concatenate(all_m, axis=0)
    d = np.concatenate(all_d, axis=0)
    ds = WellLogDataset(x, m, d)
    logger.info(f"Loader windows: {len(ds)} | x={x.shape}, m={m.shape}, d={d.shape}")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, drop_last=False)


# ============================================================
# 3. Dataset
# ============================================================
class WellLogDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, m_windows: np.ndarray, d_windows: np.ndarray):
        self.x = torch.from_numpy(x_windows).float()
        self.m_obs = torch.from_numpy(m_windows).float()
        self.depth = torch.from_numpy(d_windows).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x0 = self.x[idx].clone()
        m_obs = self.m_obs[idx].clone()
        depth = self.depth[idx].clone()
        m_ta = self.make_target_mask(m_obs)
        m_cond = m_obs * (1.0 - m_ta)
        x_cond = x0 * m_cond
        return {"x0": x0, "m_obs": m_obs, "m_ta": m_ta, "m_cond": m_cond, "x_cond": x_cond, "depth": depth}

    @staticmethod
    def make_target_mask(m_obs: torch.Tensor) -> torch.Tensor:
        c, l = m_obs.shape
        m_ta = torch.zeros_like(m_obs)
        num_ch = random.randint(MIN_TARGET_CHANNELS, min(MAX_TARGET_CHANNELS, c))
        channels = random.sample(range(c), k=num_ch)

        for ch in channels:
            seg_len = random.randint(TARGET_MIN_LEN, TARGET_MAX_LEN)
            seg_len = min(seg_len, max(1, l // 2))
            found = False
            for _ in range(50):
                start = random.randint(0, l - seg_len)
                end = start + seg_len
                # Artificial target must avoid real missing data.
                if m_obs[ch, start:end].mean() > 0.95:
                    m_ta[ch, start:end] = 1.0
                    found = True
                    break
            if not found:
                valid_idx = torch.where(m_obs[ch] > 0.5)[0]
                if len(valid_idx) > seg_len:
                    s = random.randint(0, len(valid_idx) - seg_len)
                    m_ta[ch, valid_idx[s:s + seg_len]] = 1.0
        return m_ta


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
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def group_norm(ch: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm1 = group_norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time(t_emb).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Down1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.r1 = ResBlock1D(in_ch, out_ch, time_dim)
        self.r2 = ResBlock1D(out_ch, out_ch, time_dim)
        self.down = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t):
        h = self.r2(self.r1(x, t), t)
        return self.down(h), h


class Up1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.r1 = ResBlock1D(out_ch + skip_ch, out_ch, time_dim)
        self.r2 = ResBlock1D(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.r2(self.r1(x, t), t)


class ConditionalUNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(TIME_DIM),
            nn.Linear(TIME_DIM, TIME_DIM),
            nn.SiLU(),
            nn.Linear(TIME_DIM, TIME_DIM),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv1d(2, DEPTH_CHANNELS, 1),
            nn.SiLU(),
            nn.Conv1d(DEPTH_CHANNELS, DEPTH_CHANNELS, 3, padding=1),
            nn.SiLU(),
        )
        in_ch = len(LOG_NAMES) * 3 + DEPTH_CHANNELS
        b = BASE_CHANNELS
        self.init = nn.Conv1d(in_ch, b, 3, padding=1)
        self.down1 = Down1D(b, b, TIME_DIM)
        self.down2 = Down1D(b, b * 2, TIME_DIM)
        self.down3 = Down1D(b * 2, b * 4, TIME_DIM)
        self.mid1 = ResBlock1D(b * 4, b * 4, TIME_DIM)
        self.mid2 = ResBlock1D(b * 4, b * 4, TIME_DIM)
        self.up3 = Up1D(b * 4, b * 4, b * 2, TIME_DIM)
        self.up2 = Up1D(b * 2, b * 2, b, TIME_DIM)
        self.up1 = Up1D(b, b, b, TIME_DIM)
        self.out = nn.Sequential(group_norm(b), nn.SiLU(), nn.Conv1d(b, len(LOG_NAMES), 1))

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

    @property
    def device(self):
        return next(self.parameters()).device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise, noise

    def p_losses(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = batch["x0"].to(self.device)
        x_cond = batch["x_cond"].to(self.device)
        m_cond = batch["m_cond"].to(self.device)
        m_ta = batch["m_ta"].to(self.device)
        depth = batch["depth"].to(self.device)

        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        xt, noise = self.q_sample(x0, t)
        x_ta_t = xt * m_ta
        pred_noise = self.model(x_ta_t, x_cond, m_cond, depth, t)
        return (((pred_noise - noise) ** 2) * m_ta).sum() / m_ta.sum().clamp(min=1.0)

    @torch.no_grad()
    def impute(self, x0_filled: torch.Tensor, m_obs: torch.Tensor, depth: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        self.eval()
        x0_filled = x0_filled.to(self.device)
        m_obs = m_obs.to(self.device)
        depth = depth.to(self.device)
        m_target = (1.0 - m_obs) if target_mask is None else target_mask.to(self.device)
        m_cond = 1.0 - m_target
        x_cond = x0_filled * m_cond
        x = torch.randn_like(x0_filled) * m_target

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, x_cond, m_cond, depth, t)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t))
            if step > 0:
                x = (mean + torch.sqrt(beta_t) * torch.randn_like(x)) * m_target
            else:
                x = mean * m_target
        return x_cond + x * m_target


# ============================================================
# 6. Train / test / checkpoint
# ============================================================
def checkpoint_payload(model, optimizer, epoch, train_losses, test_metrics, scaler, depth_mean, depth_std, train_files, test_files, train_times, total_time):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_losses": train_losses,
        "test_metrics": test_metrics,
        "log_scaler": scaler.state_dict(),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "train_files": [p.name for p in train_files],
        "test_files": [p.name for p in test_files],
        "train_times": train_times,
        "total_training_time": total_time,
        "config": {
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "diffusion_steps": DIFFUSION_STEPS,
            "log_names": LOG_NAMES,
        },
    }


def load_model(prefer_best: bool = True):
    path = BEST_MODEL_PATH if prefer_best and BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("No checkpoint found. Please train first.")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CDDPM().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = SimpleStandardScaler()
    scaler.load_state_dict(ckpt["log_scaler"])
    depth_mean = float(ckpt["depth_mean"])
    depth_std = float(ckpt["depth_std"])
    logger.info(f"Loaded model: {path.name}")
    return model, scaler, depth_mean, depth_std, ckpt


def train_model():
    set_seed(SEED)
    train_files, test_files = load_split_files()

    if LATEST_MODEL_PATH.exists():
        ckpt = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        scaler = SimpleStandardScaler(); scaler.load_state_dict(ckpt["log_scaler"])
        depth_mean = float(ckpt["depth_mean"]); depth_std = float(ckpt["depth_std"])
        model = CDDPM().to(DEVICE); model.load_state_dict(ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        train_losses = ckpt.get("train_losses", [])
        test_metrics = ckpt.get("test_metrics", {})
        train_times = int(ckpt.get("train_times", 0))
        total_time = float(ckpt.get("total_training_time", 0.0))
        logger.info(f"Resume training from epoch={start_epoch}")
    else:
        scaler, depth_mean, depth_std = fit_scalers(train_files)
        model = CDDPM().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        start_epoch, train_losses, test_metrics, train_times, total_time = 0, [], {}, 0, 0.0
        logger.info("Start first training.")

    train_loader = build_loader(train_files, scaler, depth_mean, depth_std, shuffle=True)
    end_epoch = start_epoch + NUM_EPOCHS
    best_loss = min(train_losses) if train_losses else float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.p_losses(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = batch["x0"].size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_loss = total_loss / max(total_n, 1)
        train_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] train_noise_loss={avg_loss:.6f}")

        elapsed = time.time() - t0
        payload = checkpoint_payload(model, optimizer, epoch + 1, train_losses, test_metrics, scaler, depth_mean, depth_std,
                                     train_files, test_files, train_times, total_time + elapsed)
        torch.save(payload, LATEST_MODEL_PATH)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(payload, BEST_MODEL_PATH)
            logger.info(f"Saved best model: {BEST_MODEL_PATH.name}")

    total_time += time.time() - t0
    payload = checkpoint_payload(model, optimizer, end_epoch, train_losses, test_metrics, scaler, depth_mean, depth_std,
                                 train_files, test_files, train_times + 1, total_time)
    torch.save(payload, LATEST_MODEL_PATH)
    logger.info(f"Training done. elapsed={total_time/60:.2f} min total")


@torch.no_grad()
def evaluate_imputation(model: CDDPM, loader: DataLoader, max_batches: int = EVAL_BATCHES) -> Dict[str, float]:
    model.eval()
    se_sum, ae_sum, n_sum = 0.0, 0.0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["x0"].to(model.device)
        m_obs = batch["m_obs"].to(model.device)
        m_ta = batch["m_ta"].to(model.device)
        depth = batch["depth"].to(model.device)
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
    _, test_files = load_split_files()
    model, scaler, depth_mean, depth_std, ckpt = load_model(prefer_best=True)
    test_loader = build_loader(test_files, scaler, depth_mean, depth_std, shuffle=False)
    metrics = evaluate_imputation(model, test_loader)
    logger.info("\n" + "=" * 55)
    logger.info(f"{'Test artificial-missing metrics':^45}")
    for k, v in metrics.items():
        logger.info(f" {k:<12}: {v:.6f}")
    logger.info("=" * 55)

    # Save metrics back to latest/best checkpoint for status display.
    ckpt["test_metrics"] = metrics
    path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    torch.save(ckpt, path)

    save_path = IMAGE_DIR / f"welllog_test_artificial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_test_example(model, test_loader, scaler, str(save_path))


# ============================================================
# 7. Impute whole well and plotting
# ============================================================
@torch.no_grad()
def impute_whole_well_once(
    model: CDDPM,
    df: pd.DataFrame,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    target_mask_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical raw logs, final completed physical logs, observed mask.

    final logs keep original observed values unchanged; only missing positions are filled.
    """
    raw_phys = df[LOG_NAMES].values.astype(np.float32)
    x, m, depth = preprocess_well_for_model(df, scaler)
    n, c = x.shape

    pred_sum = np.zeros_like(x, dtype=np.float32)
    pred_count = np.zeros_like(x, dtype=np.float32)

    starts = list(range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE)) if n >= WINDOW_SIZE else []
    if n >= WINDOW_SIZE and (not starts or starts[-1] != n - WINDOW_SIZE):
        starts.append(n - WINDOW_SIZE)
    if not starts:
        raise RuntimeError("Well is shorter than WINDOW_SIZE.")

    for start in starts:
        end = start + WINDOW_SIZE
        xw = x[start:end]
        mw = m[start:end]
        dw_abs = depth[start:end]
        rel_depth = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
        abs_depth_scaled = ((dw_abs - depth_mean) / depth_std).astype(np.float32)
        dw = np.stack([rel_depth, abs_depth_scaled], axis=0).astype(np.float32)

        if target_mask_override is not None:
            target = target_mask_override[start:end].astype(np.float32)
            if target.sum() == 0:
                continue
            target_t = torch.from_numpy(target.T[None]).float().to(model.device)
        else:
            target_t = None

        xw_t = torch.from_numpy(xw.T[None]).float().to(model.device)
        mw_t = torch.from_numpy(mw.T[None]).float().to(model.device)
        dw_t = torch.from_numpy(dw[None]).float().to(model.device)
        completed = model.impute(xw_t, mw_t, dw_t, target_mask=target_t)[0].detach().cpu().numpy().T

        if target_mask_override is not None:
            fill = target_mask_override[start:end].astype(np.float32)
        else:
            fill = 1.0 - mw
        pred_sum[start:end] += completed * fill
        pred_count[start:end] += fill

    x_completed = x.copy()
    fill_mask = pred_count > 0
    x_completed[fill_mask] = pred_sum[fill_mask] / np.maximum(pred_count[fill_mask], 1e-6)

    completed_phys_model = model_to_physical_values(x_completed, scaler)
    final_phys = raw_phys.copy()
    real_missing = ~np.isfinite(raw_phys)
    final_phys[real_missing] = completed_phys_model[real_missing]
    return raw_phys, final_phys, m


def export_imputed_csv(csv_path: Path, df: pd.DataFrame, raw_phys: np.ndarray, final_phys: np.ndarray) -> None:
    out = df.copy()
    real_missing = ~np.isfinite(raw_phys)
    for j, name in enumerate(LOG_NAMES):
        out[f"{name}_original"] = raw_phys[:, j]
        out[f"{name}_imputed"] = final_phys[:, j]
        out[name] = np.where(real_missing[:, j], final_phys[:, j], raw_phys[:, j])
    save_path = MODEL_DIR / f"{csv_path.stem}_imputed.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved imputed CSV: {save_path.name}")


def downsample_series(depth: np.ndarray, curves: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(depth)
    if n <= max_points:
        return depth, curves
    bins = np.array_split(np.arange(n), max_points)
    d_out = np.array([np.nanmean(depth[b]) for b in bins], dtype=np.float32)
    c_out = np.zeros((len(bins), curves.shape[1]), dtype=np.float32)
    for i, b in enumerate(bins):
        for j in range(curves.shape[1]):
            c_out[i, j] = np.nanmean(curves[b, j])
    return d_out, c_out


def missing_intervals(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return []
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(int(seg[0]), int(seg[-1]) + 1) for seg in splits if len(seg) > 0]


def choose_missing_interval(raw_phys: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Return (log_idx, start, end) for longest real-missing interval."""
    best = None
    for j in range(raw_phys.shape[1]):
        miss = ~np.isfinite(raw_phys[:, j])
        for s, e in missing_intervals(miss):
            if best is None or (e - s) > (best[2] - best[1]):
                best = (j, s, e)
    return best


def plot_curves(depth, observed, completed, title, save_path, max_points: Optional[int] = None, shade_missing: bool = True):
    obs = observed.copy()
    comp = completed.copy()
    if max_points is not None:
        d_plot, obs_plot = downsample_series(depth, obs, max_points)
        _, comp_plot = downsample_series(depth, comp, max_points)
    else:
        d_plot, obs_plot, comp_plot = depth, obs, comp

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 8), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(obs_plot[:, j], d_plot, label="before/observed", linewidth=1.5)
        ax.plot(comp_plot[:, j], d_plot, "--", label="after/imputed", linewidth=1.5)

        if shade_missing:
            miss = ~np.isfinite(observed[:, j])
            for s, e in missing_intervals(miss):
                ax.axhspan(depth[s], depth[e - 1], alpha=0.10)
        ax.set_title(f"{name} ({LOG_UNITS[name]})")
        ax.set_xlabel(LOG_UNITS[name])
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved figure: {Path(save_path).name}")


def plot_test_example(model: CDDPM, test_loader: DataLoader, scaler: SimpleStandardScaler, save_path: str):
    batch = next(iter(test_loader))
    x0 = batch["x0"][:1].to(model.device)
    m_obs = batch["m_obs"][:1].to(model.device)
    m_ta = batch["m_ta"][:1].to(model.device)
    depth = batch["depth"][:1].to(model.device)
    x_masked = x0 * (1.0 - m_ta)
    m_cond = m_obs * (1.0 - m_ta)
    x_hat = model.impute(x_masked, m_cond, depth, target_mask=m_ta)

    # Convert model space to physical space for plotting; use window relative depth index only for artificial test.
    original = model_to_physical_values(x0[0].detach().cpu().numpy().T, scaler)
    imputed = model_to_physical_values(x_hat[0].detach().cpu().numpy().T, scaler)
    condition_scaled = x_masked[0].detach().cpu().numpy().T
    condition_mask = m_cond[0].detach().cpu().numpy().T
    condition = model_to_physical_values(condition_scaled, scaler)
    condition[condition_mask < 0.5] = np.nan
    target = m_ta[0].detach().cpu().numpy()
    d_plot = np.arange(original.shape[0], dtype=np.float32)

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 7), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(original[:, j], d_plot, label="original")
        ax.plot(condition[:, j], d_plot, label="condition")
        ax.plot(imputed[:, j], d_plot, "--", label="imputed")
        idx = np.where(target[j] > 0.5)[0]
        if len(idx) > 0:
            ax.axhspan(idx.min(), idx.max(), alpha=0.15)
        ax.set_title(f"{name} ({LOG_UNITS[name]})")
        ax.set_xlabel(LOG_UNITS[name])
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    axes[0].set_ylabel("Window depth index")
    axes[-1].legend()
    fig.suptitle("Artificial missing test example")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved test example: {Path(save_path).name}")


@torch.no_grad()
def plot_zoom_and_uncertainty(csv_path: Path, model: CDDPM, scaler: SimpleStandardScaler, depth_mean: float, depth_std: float):
    df = read_one_csv(csv_path)
    raw_phys, final_phys, _ = impute_whole_well_once(model, df, scaler, depth_mean, depth_std)
    depth = df["Depth"].values.astype(np.float32)
    export_imputed_csv(csv_path, df, raw_phys, final_phys)

    # Whole-well overview.
    overview_path = CASE_DIR / f"{csv_path.stem}_overview_whole_well.png"
    plot_curves(depth, raw_phys, final_phys, f"Whole-well overview: {csv_path.name}", str(overview_path), max_points=OVERVIEW_POINTS)

    # Select longest real-missing interval for zoom and uncertainty.
    best = choose_missing_interval(raw_phys)
    if best is None:
        logger.info(f"[CASE] {csv_path.name} has no real missing interval; only overview exported.")
        return
    log_idx, s, e = best
    z0 = max(0, s - ZOOM_CONTEXT_POINTS)
    z1 = min(len(depth), e + ZOOM_CONTEXT_POINTS)

    zoom_path = CASE_DIR / f"{csv_path.stem}_zoom_missing_interval.png"
    plot_curves(
        depth[z0:z1],
        raw_phys[z0:z1],
        final_phys[z0:z1],
        f"Zoom missing interval: {csv_path.name} | {LOG_NAMES[log_idx]} | {depth[s]:.2f}-{depth[e-1]:.2f} m",
        str(zoom_path),
        max_points=None,
    )

    # Local multi-sample uncertainty: only re-sample a local segment, not the whole well.
    plot_local_uncertainty(csv_path, df, model, scaler, depth_mean, depth_std, log_idx, z0, z1, s, e)


@torch.no_grad()
def plot_local_uncertainty(csv_path, df, model, scaler, depth_mean, depth_std, log_idx, z0, z1, miss_s, miss_e):
    depth = df["Depth"].values.astype(np.float32)
    raw_phys = df[LOG_NAMES].values.astype(np.float32)

    # Build a local target override in model-space window procedure: only target true missing within [z0,z1].
    target_override = np.zeros((len(df), len(LOG_NAMES)), dtype=np.float32)
    target_override[miss_s:miss_e, log_idx] = 1.0

    samples = []
    for k in range(LOCAL_MC_SAMPLES):
        _, final_phys, _ = impute_whole_well_once(model, df, scaler, depth_mean, depth_std, target_mask_override=target_override)
        samples.append(final_phys[z0:z1, log_idx])
        logger.info(f"  local MC sample {k + 1}/{LOCAL_MC_SAMPLES} done")
    samples = np.stack(samples, axis=0)
    mean_curve = np.nanmean(samples, axis=0)

    d = depth[z0:z1]
    obs = raw_phys[z0:z1, log_idx].copy()
    obs[~np.isfinite(obs)] = np.nan

    plt.figure(figsize=(6, 8))
    plt.plot(obs, d, color="black", linewidth=2, label="observed")
    for k in range(LOCAL_MC_SAMPLES):
        plt.plot(samples[k], d, linestyle="--", linewidth=1.2, alpha=0.8, label=f"sample {k+1}")
    plt.plot(mean_curve, d, linewidth=2.5, label="mean")
    plt.axhspan(depth[miss_s], depth[miss_e - 1], alpha=0.12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    name = LOG_NAMES[log_idx]
    plt.xlabel(f"{name} ({LOG_UNITS[name]})")
    plt.ylabel(f"Depth ({LOG_UNITS['Depth']})")
    plt.title(f"Local uncertainty: {csv_path.name} | {name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_path = CASE_DIR / f"{csv_path.stem}_local_uncertainty_multi_samples.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved uncertainty figure: {save_path.name}")


# ============================================================
# 8. Modes / menu
# ============================================================
def generate_cases():
    _, test_files = load_split_files()
    model, scaler, depth_mean, depth_std, _ = load_model(prefer_best=True)
    num = input("请输入案例井数量 1~3，默认 3: ").strip()
    try:
        num_cases = min(max(int(num), 1), 3) if num else 3
    except Exception:
        num_cases = 3
    selected = test_files[:num_cases]
    logger.info(f"Selected cases: {[p.name for p in selected]}")
    for p in selected:
        try:
            plot_zoom_and_uncertainty(p, model, scaler, depth_mean, depth_std)
        except Exception as e:
            logger.error(f"Generate case failed: {p.name}: {e}")


def generate_one_well():
    user_input = input("请输入 CSV 文件名/stem/完整路径: ").strip()
    if not user_input:
        return
    all_files = scan_csv_files(CSV_DIR)
    name_map = {}
    for p in all_files:
        name_map[p.name] = p
        name_map[p.stem] = p
    csv_path = Path(user_input)
    if not csv_path.exists():
        csv_path = name_map.get(user_input)
    if csv_path is None or not Path(csv_path).exists():
        logger.error(f"找不到 CSV: {user_input}")
        return
    model, scaler, depth_mean, depth_std, _ = load_model(prefer_best=True)
    plot_zoom_and_uncertainty(Path(csv_path), model, scaler, depth_mean, depth_std)


def show_model_status():
    path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    if not path.exists():
        logger.info("尚未发现模型文件。")
        return
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    logger.info("\n" + "=" * 60)
    logger.info(f"{'WellLog CDDPM U-Net 状态':^50}")
    logger.info(f"模型文件: {path.name}")
    logger.info(f"已训练 epoch: {ckpt.get('epoch', 0)}")
    logger.info(f"累计训练次数: {ckpt.get('train_times', 0)}")
    losses = ckpt.get("train_losses", [])
    if losses:
        logger.info(f"最近训练 loss: {losses[-1]:.6f}")
        logger.info(f"最优训练 loss: {min(losses):.6f}")
    metrics = ckpt.get("test_metrics", {})
    if metrics:
        logger.info("最近测试指标:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.6f}")
    else:
        logger.info("最近测试指标: 尚未执行 test")
    logger.info(f"训练总时长: {ckpt.get('total_training_time', 0.0)/60:.2f} min")
    logger.info(f"CSV_DIR: {CSV_DIR}")
    logger.info("=" * 60)


def generate_loss_plot():
    path = LATEST_MODEL_PATH if LATEST_MODEL_PATH.exists() else BEST_MODEL_PATH
    if not path.exists():
        logger.info("尚未发现模型文件。")
        return
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    losses = ckpt.get("train_losses", [])
    if not losses:
        logger.info("checkpoint 中没有 train_losses。")
        return
    epochs = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Noise prediction MSE")
    plt.title(f"WellLog CDDPM Training Loss (epochs={len(losses)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()
    logger.info(f"Saved loss plot: {LOSS_PLOT_PATH.name}")


def main():
    set_seed(SEED)
    logger.info(f"Device: {DEVICE}")
    while True:
        logger.info("\n" + "=" * 55)
        logger.info("        WellLog CDDPM U-Net 管理系统")
        logger.info(" [1] 训练模型 Train")
        logger.info(" [2] 测试指标 Test Metrics")
        logger.info(" [3] 生成案例 Generate Cases")
        logger.info(" [4] 指定单井 Generate One Well")
        logger.info(" [5] 查看模型状态 Status")
        logger.info(" [6] 生成损失函数图 Loss Plot")
        logger.info(" [0/exit] 退出")
        logger.info("=" * 55)
        choice = input("请选择功能: ").strip().lower()
        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            generate_cases()
        elif choice == "4":
            generate_one_well()
        elif choice == "5":
            show_model_status()
        elif choice == "6":
            generate_loss_plot()
        elif choice in ("0", "exit"):
            logger.info("退出程序。")
            break
        else:
            logger.info("无效输入。")


if __name__ == "__main__":
    main()
