"""
CDDPM Well-log Imputation with 1D Conditional ResNet

Required CSV columns:
    Depth, GR, RHOB, RILD, CNPOR

Main points:
    1. Use existing npz split file: train/test or train/val/test.
       If val exists, val files are merged into train.
    2. Train with artificial masking only on originally observed values.
    3. Test by artificial masking on held-out test wells and compute MAE/RMSE/MSE.
    4. Generate case figures in a timestamped run directory:
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
from sklearn.metrics import r2_score

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
NUM_EPOCHS = 50
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
CONDITION_MC_SAMPLES = 5   # Monte Carlo samples for STRONG/WEAK artificial cases

LATEST_MODEL_PATH = MODEL_DIR / "latest_welllog_cddpm_resnet.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_welllog_cddpm_resnet.pth"
LOSS_PLOT_PATH = MODEL_DIR / "welllog_cddpm_resnet_loss.png"
CASE_DIR = IMAGE_DIR / "welllog_case_figures"
CASE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = IMAGE_DIR / "welllog_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(tag: str) -> Path:
    """Create a timestamped output directory for one operation.

    This avoids overwriting previous figures/CSV files and groups all outputs
    from the same run together.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = tag.replace(" ", "_")
    run_dir = RUNS_DIR / f"{ts}_{safe_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
# 4. 1D Conditional ResNet
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


class ConditionalResNet1D(nn.Module):
    """
    1D Conditional ResNet denoiser for CDDPM.

    Compared with the U-Net version, this network does not downsample/upsample.
    It is faster and simpler, and is suitable for short- to medium-length missing
    segments where local context is already enough.

    Inputs:
        x_ta_t: [B, C, L] noisy target region, non-target region is zero
        x_cond: [B, C, L] visible condition logs, hidden/missing region is zero
        m_cond: [B, C, L] condition mask, 1=visible, 0=invisible
        depth:  [B, 2, L] relative depth + standardized absolute depth
        t:      [B]
    Output:
        pred_noise: [B, C, L]
    """
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
        self.input = nn.Conv1d(in_ch, b, 3, padding=1)

        # Standard residual blocks. Dilation is added to enlarge receptive field
        # while keeping the sequence length unchanged.
        self.blocks = nn.ModuleList([
            ResBlock1D(b, b, TIME_DIM),
            ResBlock1D(b, b, TIME_DIM),
            ResBlock1D(b, b, TIME_DIM),
            ResBlock1D(b, b, TIME_DIM),
            ResBlock1D(b, b, TIME_DIM),
            ResBlock1D(b, b, TIME_DIM),
        ])

        self.out = nn.Sequential(
            group_norm(b),
            nn.SiLU(),
            nn.Conv1d(b, b, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(b, len(LOG_NAMES), 1),
        )

    def forward(self, x_ta_t, x_cond, m_cond, depth, t):
        t_emb = self.time_emb(t)
        d_emb = self.depth_proj(depth)
        x = torch.cat([x_ta_t, x_cond, m_cond, d_emb], dim=1)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)


# ============================================================
# 5. CDDPM
# ============================================================
class CDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ConditionalResNet1D()
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
        # Only originally observed values that are not target can be used as condition.
        # This prevents real-missing points from being treated as visible condition when
        # target_mask is provided for artificial-missing examples.
        m_cond = m_obs * (1.0 - m_target)
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


    @torch.no_grad()
    def impute_with_snapshots(
        self,
        x0_filled: torch.Tensor,
        m_obs: torch.Tensor,
        depth: torch.Tensor,
        target_mask: torch.Tensor,
        snapshot_steps: Optional[List[int]] = None,
    ):
        """Impute and keep intermediate reverse-denoising states.

        snapshot_steps are diffusion step indices, e.g. [199, 150, 100, 50, 0].
        Returned snapshots are completed windows: condition part + current target part.
        """
        self.eval()
        if snapshot_steps is None:
            snapshot_steps = [self.timesteps - 1, int(self.timesteps * 0.75), int(self.timesteps * 0.50), int(self.timesteps * 0.25), 0]
        snapshot_steps = sorted(set([max(0, min(self.timesteps - 1, int(t))) for t in snapshot_steps]), reverse=True)

        x0_filled = x0_filled.to(self.device)
        m_obs = m_obs.to(self.device)
        depth = depth.to(self.device)
        m_target = target_mask.to(self.device)
        # Same rule as impute(): condition = originally observed AND not target.
        m_cond = m_obs * (1.0 - m_target)
        x_cond = x0_filled * m_cond
        x = torch.randn_like(x0_filled) * m_target

        snapshots = {}
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

            if step in snapshot_steps:
                snapshots[step] = (x_cond + x * m_target).detach().cpu()

        final = (x_cond + x * m_target).detach().cpu()
        return final, snapshots


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
def evaluate_imputation(
    model: CDDPM,
    loader: DataLoader,
    max_batches: int = EVAL_BATCHES
) -> Dict[str, float]:

    model.eval()

    se_sum = 0.0
    ae_sum = 0.0
    n_sum = 0.0

    # 用于 R²
    all_gt = []
    all_pred = []

    for i, batch in enumerate(loader):

        if i >= max_batches:
            break

        x0 = batch["x0"].to(model.device)
        m_obs = batch["m_obs"].to(model.device)
        m_ta = batch["m_ta"].to(model.device)
        depth = batch["depth"].to(model.device)

        # 人工挖缺
        x_masked = x0 * (1.0 - m_ta)
        m_cond = m_obs * (1.0 - m_ta)

        # 补全
        x_hat = model.impute(
            x_masked,
            m_cond,
            depth,
            target_mask=m_ta
        )

        # 只统计 target 区域
        diff = (x_hat - x0) * m_ta

        se_sum += float((diff ** 2).sum().item())
        ae_sum += float(diff.abs().sum().item())
        n_sum += float(m_ta.sum().item())

        # ========= R² =========
        gt = x0[m_ta > 0.5].detach().cpu().numpy()
        pred = x_hat[m_ta > 0.5].detach().cpu().numpy()

        if len(gt) > 0:
            all_gt.append(gt)
            all_pred.append(pred)

    mse = se_sum / max(n_sum, 1.0)
    rmse = math.sqrt(mse)
    mae = ae_sum / max(n_sum, 1.0)

    # ========= 计算 R² =========
    if len(all_gt) > 0:
        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        try:
            r2 = r2_score(all_gt, all_pred)
        except Exception:
            r2 = float("nan")
    else:
        r2 = float("nan")

    return {
        "MAE_std": mae,
        "RMSE_std": rmse,
        "MSE_std": mse,
        "R2_std": r2,
    }


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

    run_dir = make_run_dir("test_resnet")

    # 每次随机选择一口测试井来画人工缺失补全示例，避免总是同一口井。
    rng = random.Random()
    candidate_files = test_files.copy()
    rng.shuffle(candidate_files)

    chosen = None
    for p in candidate_files:
        try:
            # 先尝试画图；如果这口井无法构造有效窗口，就换下一口井。
            save_path = run_dir / f"welllog_test_artificial_{p.stem}.png"
            plot_artificial_test_example_from_file(model, p, scaler, depth_mean, depth_std, str(save_path))
            chosen = p
            break
        except Exception as e:
            logger.info(f"[TEST-FIG-SKIP] {p.name}: {e}")

    # GR 逐步去噪流程图：5 个时间步，从随机噪声逐步恢复到清晰曲线。
    denoise_chosen = None
    for p in candidate_files:
        try:
            denoise_path = run_dir / f"gr_denoising_process_{p.stem}.png"
            plot_gr_denoising_process_from_file(model, p, scaler, depth_mean, depth_std, str(denoise_path))
            denoise_chosen = p
            break
        except Exception as e:
            logger.info(f"[DENOISE-FIG-SKIP] {p.name}: {e}")

    if chosen is not None:
        logger.info(f"Random test example well: {chosen.name}")
    else:
        logger.warning("No valid test well found for artificial-missing figure.")

    if denoise_chosen is not None:
        logger.info(f"GR denoising process well: {denoise_chosen.name}")
    else:
        logger.warning("No valid test well found for GR denoising process figure.")
    logger.info(f"Test figures saved in: {run_dir}")


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


def export_imputed_csv(csv_path: Path, df: pd.DataFrame, raw_phys: np.ndarray, final_phys: np.ndarray, output_dir: Path) -> None:
    out = df.copy()
    real_missing = ~np.isfinite(raw_phys)
    for j, name in enumerate(LOG_NAMES):
        out[f"{name}_original"] = raw_phys[:, j]
        out[f"{name}_imputed"] = final_phys[:, j]
        out[name] = np.where(real_missing[:, j], final_phys[:, j], raw_phys[:, j])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{csv_path.stem}_imputed.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved imputed CSV: {save_path}")


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


def depth_range(depth: np.ndarray) -> Tuple[float, float]:
    """Return original measured-depth range without any relative-depth conversion."""
    depth = np.asarray(depth, dtype=np.float32)
    if len(depth) == 0:
        return float("nan"), float("nan")
    return float(np.nanmin(depth)), float(np.nanmax(depth))


def format_depth_title(prefix: str, file_name: str, top: float, bottom: float) -> str:
    return f"{prefix}: {file_name} | Depth range: {top:.2f}-{bottom:.2f} {LOG_UNITS['Depth']}"


def apply_log_depth_style(ax, depth: np.ndarray) -> None:
    """Standard well-log depth axis: original Depth values, increasing downward."""
    depth = np.asarray(depth, dtype=np.float32)
    if len(depth) > 0:
        dmin = float(np.nanmin(depth))
        dmax = float(np.nanmax(depth))
        ax.set_ylim(dmax, dmin)  # inverted y-axis: shallow/top to deep/bottom
    ax.grid(True, alpha=0.3)


def plot_curves(depth, observed, completed, title, save_path, max_points: Optional[int] = None, shade_missing: bool = True):
    """Plot observed vs imputed curves with unified well-log style.

    Important:
        - y-axis uses the original CSV Depth values directly.
        - y-axis is inverted so depth increases downward.
        - Existing observed values are never changed by plotting.
    """
    obs = observed.copy()
    comp = completed.copy()
    depth = np.asarray(depth, dtype=np.float32)

    if max_points is not None:
        d_plot, obs_plot = downsample_series(depth, obs, max_points)
        _, comp_plot = downsample_series(depth, comp, max_points)
    else:
        d_plot, obs_plot, comp_plot = depth, obs, comp

    top, bottom = depth_range(depth)

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 8), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(obs_plot[:, j], d_plot, label="observed", linewidth=1.5)
        ax.plot(comp_plot[:, j], d_plot, "--", label="imputed", linewidth=1.5)

        if shade_missing:
            miss = ~np.isfinite(observed[:, j])
            for s, e in missing_intervals(miss):
                ax.axhspan(float(depth[s]), float(depth[e - 1]), alpha=0.10)
        ax.set_title(name)
        ax.set_xlabel(LOG_UNITS[name])
        apply_log_depth_style(ax, d_plot)
    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend()
    fig.suptitle(format_depth_title(title, "", top, bottom).replace(":  |", " |"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved figure: {Path(save_path).name}")


@torch.no_grad()
def plot_artificial_test_example_from_file(
    model: CDDPM,
    csv_path: Path,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    save_path: str,
):
    """Plot one artificial-missing test window with file name and real depth range.

    This is clearer than plotting a random DataLoader window because the figure can
    show the source file name and the true measured-depth interval.
    """
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_well_for_model(df, scaler)
    xw, mw, dw, spans = make_windows(x, m, depth, depth_mean, depth_std)
    if len(xw) == 0:
        logger.info(f"No valid test window for {csv_path.name}")
        return

    ds = WellLogDataset(xw, mw, dw)
    sample = None
    sample_idx = 0
    for i in range(min(len(ds), 50)):
        cand = ds[i]
        if cand["m_ta"].sum() > 0:
            sample = cand
            sample_idx = i
            break
    if sample is None:
        logger.info(f"Cannot create artificial target for {csv_path.name}")
        return

    start, end = spans[sample_idx]
    true_depth = depth[start:end]
    top, bottom = depth_range(true_depth)

    x0 = sample["x0"][None].to(model.device)
    m_obs = sample["m_obs"][None].to(model.device)
    m_ta = sample["m_ta"][None].to(model.device)
    depth_t = sample["depth"][None].to(model.device)

    x_masked = x0 * (1.0 - m_ta)
    m_cond = m_obs * (1.0 - m_ta)
    x_hat = model.impute(x_masked, m_cond, depth_t, target_mask=m_ta)

    original = model_to_physical_values(x0[0].detach().cpu().numpy().T, scaler)
    imputed = model_to_physical_values(x_hat[0].detach().cpu().numpy().T, scaler)
    condition = model_to_physical_values(x_masked[0].detach().cpu().numpy().T, scaler)
    condition_mask = m_cond[0].detach().cpu().numpy().T
    condition[condition_mask < 0.5] = np.nan
    target = m_ta[0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 7), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(original[:, j], true_depth, label="original")
        ax.plot(condition[:, j], true_depth, label="condition")
        ax.plot(imputed[:, j], true_depth, "--", label="imputed")
        idx = np.where(target[j] > 0.5)[0]
        if len(idx) > 0:
            ax.axhspan(float(true_depth[idx.min()]), float(true_depth[idx.max()]), alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel(LOG_UNITS[name])
        apply_log_depth_style(ax, true_depth)
    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend()
    fig.suptitle(format_depth_title("Artificial missing test", csv_path.name, top, bottom))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved test example: {Path(save_path).name}")




# ============================================================
# 7.1 Extra visualization: denoising process and condition strength
# ============================================================
def make_gr_segment_mask(m_obs_window: np.ndarray, num_segments: int, min_len: int, max_len: int, rng: random.Random) -> Optional[np.ndarray]:
    """Create an artificial target mask on GR only.

    Args:
        m_obs_window: [C, L], 1 means originally observed.
        num_segments: number of GR intervals to remove.
    Returns:
        mask [C, L], or None if no valid intervals can be found.
    """
    c, l = m_obs_window.shape
    gr_idx = LOG_NAMES.index("GR")
    target = np.zeros((c, l), dtype=np.float32)
    occupied = np.zeros(l, dtype=bool)

    for _ in range(num_segments):
        found = False
        seg_len = rng.randint(min_len, max_len)
        seg_len = min(seg_len, max(1, l // 3))
        for _try in range(100):
            start = rng.randint(0, l - seg_len)
            end = start + seg_len
            # Require observed GR and avoid overlapping/too adjacent intervals.
            pad_s = max(0, start - 4)
            pad_e = min(l, end + 4)
            if occupied[pad_s:pad_e].any():
                continue
            if m_obs_window[gr_idx, start:end].mean() > 0.98:
                target[gr_idx, start:end] = 1.0
                occupied[pad_s:pad_e] = True
                found = True
                break
        if not found:
            return None
    return target


def build_artificial_gr_sample_from_file(
    csv_path: Path,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    num_segments: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
):
    """Pick a window from one well and create GR-only artificial missing segments."""
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_well_for_model(df, scaler)
    xw, mw, dw, spans = make_windows(x, m, depth, depth_mean, depth_std)
    if len(xw) == 0:
        return None

    indices = list(range(len(xw)))
    rng.shuffle(indices)
    for idx in indices[:80]:
        target = make_gr_segment_mask(mw[idx], num_segments, min_len, max_len, rng)
        if target is None or target.sum() == 0:
            continue
        start, end = spans[idx]
        return {
            "csv_path": csv_path,
            "window_idx": idx,
            "span": (start, end),
            "depth": depth[start:end],
            "x0": torch.from_numpy(xw[idx][None]).float(),
            "m_obs": torch.from_numpy(mw[idx][None]).float(),
            "dw": torch.from_numpy(dw[idx][None]).float(),
            "target": torch.from_numpy(target[None]).float(),
        }
    return None


@torch.no_grad()
def plot_gr_denoising_process_from_file(
    model: CDDPM,
    csv_path: Path,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    save_path: str,
):
    """Save a 5-step GR reverse-denoising process figure for Test Metrics."""
    rng = random.Random(SEED + 2026)
    sample = build_artificial_gr_sample_from_file(
        csv_path, scaler, depth_mean, depth_std,
        num_segments=1, min_len=max(8, TARGET_MIN_LEN), max_len=min(40, TARGET_MAX_LEN), rng=rng,
    )
    if sample is None:
        raise RuntimeError(f"No valid GR artificial-missing sample in {csv_path.name}")

    snapshot_steps = [model.timesteps - 1, int(model.timesteps * 0.75), int(model.timesteps * 0.50), int(model.timesteps * 0.25), 0]
    x0 = sample["x0"].to(model.device)
    m_obs = sample["m_obs"].to(model.device)
    target = sample["target"].to(model.device)
    depth_t = sample["dw"].to(model.device)
    x_masked = x0 * (1.0 - target)
    m_cond = m_obs * (1.0 - target)

    _, snapshots = model.impute_with_snapshots(x_masked, m_cond, depth_t, target, snapshot_steps=snapshot_steps)

    gr_idx = LOG_NAMES.index("GR")
    d_true = sample["depth"].astype(np.float32)
    original_phys = model_to_physical_values(x0[0].detach().cpu().numpy().T, scaler)[:, gr_idx]
    condition_scaled = x_masked[0].detach().cpu().numpy().T
    condition_phys = model_to_physical_values(condition_scaled, scaler)[:, gr_idx]
    condition_mask = m_cond[0, gr_idx].detach().cpu().numpy()
    condition_phys[condition_mask < 0.5] = np.nan
    target_1d = target[0, gr_idx].detach().cpu().numpy()
    miss_idx = np.where(target_1d > 0.5)[0]

    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
    for ax, step in zip(axes, snapshot_steps):
        snap = snapshots[int(step)][0].numpy().T
        snap_phys = model_to_physical_values(snap, scaler)[:, gr_idx]
        ax.plot(original_phys, d_true, linewidth=1.5, label="original")
        ax.plot(condition_phys, d_true, linewidth=1.5, label="condition")
        ax.plot(snap_phys, d_true, "--", linewidth=1.8, label="denoised GR")
        if len(miss_idx) > 0:
            ax.axhspan(float(d_true[miss_idx.min()]), float(d_true[miss_idx.max()]), alpha=0.15)
        ax.set_title(f"t = {int(step)}")
        ax.set_xlabel(LOG_UNITS["GR"])
        apply_log_depth_style(ax, d_true)
    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend(fontsize=8, loc="best")
    top, bottom = depth_range(d_true)
    fig.suptitle(format_depth_title("GR reverse denoising process | noise → clear", sample["csv_path"].name, top, bottom))
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    logger.info(f"Saved GR denoising process figure: {Path(save_path).name}")


@torch.no_grad()
def plot_condition_strength_sample(
    sample: Dict,
    model: CDDPM,
    scaler: SimpleStandardScaler,
    save_path: Path,
    title_prefix: str,
    num_samples: int = CONDITION_MC_SAMPLES,
):
    """Plot STRONG/WEAK artificial sampling case with multiple diffusion samples.

    The target mask is fixed. The model is sampled several times from different
    initial noises. This is the key visualization for conditional uncertainty:
        - strong condition: sample curves should be closer
        - weak condition: sample curves are usually more dispersed
    """
    x0 = sample["x0"].to(model.device)
    m_obs = sample["m_obs"].to(model.device)
    target = sample["target"].to(model.device)
    depth_t = sample["dw"].to(model.device)
    true_depth = sample["depth"].astype(np.float32)

    x_masked = x0 * (1.0 - target)
    m_cond = m_obs * (1.0 - target)

    # Multiple Monte Carlo samples under the same condition and the same target mask.
    # Do NOT reset random seed here; otherwise the five samples may become identical.
    sample_phys_list = []
    for k in range(num_samples):
        # Pass m_obs, not m_cond. impute() internally builds:
        #   m_cond = m_obs * (1 - target_mask)
        x_hat = model.impute(x_masked, m_obs, depth_t, target_mask=target)
        sample_phys = model_to_physical_values(x_hat[0].detach().cpu().numpy().T, scaler)
        sample_phys_list.append(sample_phys)
        logger.info(f"  {title_prefix} MC sample {k + 1}/{num_samples} done")

    samples = np.stack(sample_phys_list, axis=0)  # [S, L, C]
    mean_curve = np.nanmean(samples, axis=0)      # [L, C]

    original = model_to_physical_values(x0[0].detach().cpu().numpy().T, scaler)
    condition = model_to_physical_values(x_masked[0].detach().cpu().numpy().T, scaler)
    condition_mask = m_cond[0].detach().cpu().numpy().T
    condition[condition_mask < 0.5] = np.nan
    target_np = target[0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(16, 7), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(original[:, j], true_depth, color="black", linewidth=1.8, label="original")
        ax.plot(condition[:, j], true_depth, color="gray", linewidth=1.5, label="condition")

        for k in range(num_samples):
            ax.plot(
                samples[k, :, j],
                true_depth,
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
                label=f"sample {k + 1}" if j == len(LOG_NAMES) - 1 else None,
            )

        ax.plot(
            mean_curve[:, j],
            true_depth,
            linewidth=2.2,
            label="mean" if j == len(LOG_NAMES) - 1 else None,
        )

        idx = np.where(target_np[j] > 0.5)[0]
        if len(idx) > 0:
            groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for g in groups:
                if len(g) > 0:
                    ax.axhspan(float(true_depth[g.min()]), float(true_depth[g.max()]), alpha=0.15)
        ax.set_title(name)
        ax.set_xlabel(LOG_UNITS[name])
        apply_log_depth_style(ax, true_depth)

    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend(fontsize=8, loc="best")
    top, bottom = depth_range(true_depth)
    fig.suptitle(format_depth_title(f"{title_prefix} | {num_samples} MC samples", sample["csv_path"].name, top, bottom))
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    logger.info(f"Saved condition-strength multi-sample case: {save_path.name}")


def build_condition_strength_pair_from_file(
    csv_path: Path,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    rng: random.Random,
):
    """Build STRONG and WEAK GR masks on the same well window.

    Using the same window is important: it keeps the geological background fixed.
    The only difference between STRONG and WEAK is how much GR is hidden.
    """
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_well_for_model(df, scaler)
    xw, mw, dw, spans = make_windows(x, m, depth, depth_mean, depth_std)
    if len(xw) == 0:
        return None, None

    indices = list(range(len(xw)))
    rng.shuffle(indices)
    for idx in indices[:120]:
        strong_target = make_gr_segment_mask(mw[idx], num_segments=1, min_len=12, max_len=24, rng=rng)
        weak_segments = rng.choice([2, 3])
        weak_target = make_gr_segment_mask(mw[idx], num_segments=weak_segments, min_len=14, max_len=30, rng=rng)
        if strong_target is None or weak_target is None:
            continue
        if strong_target.sum() == 0 or weak_target.sum() == 0:
            continue

        start, end = spans[idx]
        base = {
            "csv_path": csv_path,
            "window_idx": idx,
            "span": (start, end),
            "depth": depth[start:end],
            "x0": torch.from_numpy(xw[idx][None]).float(),
            "m_obs": torch.from_numpy(mw[idx][None]).float(),
            "dw": torch.from_numpy(dw[idx][None]).float(),
        }
        strong_sample = dict(base)
        weak_sample = dict(base)
        strong_sample["target"] = torch.from_numpy(strong_target[None]).float()
        weak_sample["target"] = torch.from_numpy(weak_target[None]).float()
        return strong_sample, weak_sample

    return None, None


@torch.no_grad()
def generate_condition_strength_cases_for_file(
    csv_path: Path,
    model: CDDPM,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    output_dir: Path,
    rng: random.Random,
):
    """Generate STRONG-vs-WEAK artificial GR cases for one selected well.

    Each selected well gets two figures from the same window:
        - STRONG: one short missing segment
        - WEAK: two or three missing segments

    Each figure contains 5 Monte Carlo sampling curves plus their mean curve.
    """
    strong_sample, weak_sample = build_condition_strength_pair_from_file(
        csv_path, scaler, depth_mean, depth_std, rng
    )

    if strong_sample is not None:
        s0, s1 = strong_sample["span"]
        plot_condition_strength_sample(
            strong_sample, model, scaler,
            output_dir / f"condition_strength_STRONG_{csv_path.stem}_win{strong_sample['window_idx']}_idx{s0}_{s1}_mc{CONDITION_MC_SAMPLES}.png",
            "Strong condition sampling | GR one short missing segment",
            num_samples=CONDITION_MC_SAMPLES,
        )
    else:
        logger.warning(f"No valid strong-condition GR sample found for {csv_path.name}.")

    if weak_sample is not None:
        num_seg = len(missing_intervals(weak_sample["target"][0, LOG_NAMES.index("GR")].numpy() > 0.5))
        s0, s1 = weak_sample["span"]
        plot_condition_strength_sample(
            weak_sample, model, scaler,
            output_dir / f"condition_strength_WEAK_{csv_path.stem}_win{weak_sample['window_idx']}_idx{s0}_{s1}_mc{CONDITION_MC_SAMPLES}.png",
            f"Weak condition sampling | GR {num_seg} missing segments",
            num_samples=CONDITION_MC_SAMPLES,
        )
    else:
        logger.warning(f"No valid weak-condition GR sample found for {csv_path.name}.")

@torch.no_grad()
def plot_zoom_and_uncertainty(csv_path: Path, model: CDDPM, scaler: SimpleStandardScaler, depth_mean: float, depth_std: float, output_dir: Path):
    df = read_one_csv(csv_path)
    raw_phys, final_phys, _ = impute_whole_well_once(model, df, scaler, depth_mean, depth_std)
    depth = df["Depth"].values.astype(np.float32)
    export_imputed_csv(csv_path, df, raw_phys, final_phys, output_dir)

    # Whole-well overview.
    overview_path = output_dir / f"{csv_path.stem}_overview_whole_well.png"
    plot_curves(depth, raw_phys, final_phys, f"Whole-well overview: {csv_path.name}", str(overview_path), max_points=OVERVIEW_POINTS)

    # Select longest real-missing interval for zoom and uncertainty.
    best = choose_missing_interval(raw_phys)
    if best is None:
        logger.info(f"[CASE] {csv_path.name} has no real missing interval; only overview exported.")
        return
    log_idx, s, e = best
    z0 = max(0, s - ZOOM_CONTEXT_POINTS)
    z1 = min(len(depth), e + ZOOM_CONTEXT_POINTS)

    zoom_path = output_dir / f"{csv_path.stem}_zoom_missing_interval.png"
    plot_curves(
        depth[z0:z1],
        raw_phys[z0:z1],
        final_phys[z0:z1],
        f"Zoom missing interval: {csv_path.name} | {LOG_NAMES[log_idx]} | {depth[s]:.2f}-{depth[e-1]:.2f} m",
        str(zoom_path),
        max_points=None,
    )

    # Local multi-sample uncertainty: only re-sample a local segment, not the whole well.
    plot_local_uncertainty(csv_path, df, model, scaler, depth_mean, depth_std, log_idx, z0, z1, s, e, output_dir)


@torch.no_grad()
def plot_local_uncertainty(csv_path, df, model, scaler, depth_mean, depth_std, log_idx, z0, z1, miss_s, miss_e, output_dir: Path):
    """Plot local multi-sample uncertainty for all log curves.

    这里不再只画某一条曲线，而是在同一个局部深度段内同时画：
        GR / RHOB / RILD / CNPOR
    每个子图中都包含：
        observed 原始观测曲线
        sample 1~N 多次采样补全结果
        mean 多次采样均值
        shaded area 当前曲线在该局部段内的真实缺失区间

    注意：
    - 多次采样只针对当前局部窗口内的真实缺失位置，不对整口井重复采样。
    - 如果某条曲线在该局部段没有真实缺失，那么 sample 曲线会基本与 observed 重合，这是正常的，表示该曲线没有被补全。
    """
    depth = df["Depth"].values.astype(np.float32)
    raw_phys = df[LOG_NAMES].values.astype(np.float32)

    # 只对局部段内“真实缺失”的位置做多次补全；四条曲线都包含在内。
    # shape: [N, C], 1 表示这个位置需要作为 target 重新采样。
    target_override = np.zeros((len(df), len(LOG_NAMES)), dtype=np.float32)
    local_real_missing = ~np.isfinite(raw_phys[z0:z1])
    target_override[z0:z1] = local_real_missing.astype(np.float32)

    if target_override.sum() == 0:
        logger.info(f"[UNCERTAINTY-SKIP] {csv_path.name}: selected local interval has no real missing values.")
        return

    samples = []
    for k in range(LOCAL_MC_SAMPLES):
        _, final_phys, _ = impute_whole_well_once(
            model,
            df,
            scaler,
            depth_mean,
            depth_std,
            target_mask_override=target_override,
        )
        samples.append(final_phys[z0:z1])  # [local_len, C]
        logger.info(f"  local MC sample {k + 1}/{LOCAL_MC_SAMPLES} done")

    samples = np.stack(samples, axis=0)  # [S, local_len, C]
    mean_curves = np.nanmean(samples, axis=0)  # [local_len, C]

    d_true = depth[z0:z1]
    top, bottom = depth_range(d_true)

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(16, 8), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]

        obs = raw_phys[z0:z1, j].copy()
        obs[~np.isfinite(obs)] = np.nan
        ax.plot(obs, d_true, color="black", linewidth=2.0, label="observed")

        for k in range(LOCAL_MC_SAMPLES):
            ax.plot(
                samples[k, :, j],
                d_true,
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
                label=f"sample {k + 1}" if j == len(LOG_NAMES) - 1 else None,
            )

        ax.plot(
            mean_curves[:, j],
            d_true,
            linewidth=2.2,
            label="mean" if j == len(LOG_NAMES) - 1 else None,
        )

        # 给每条曲线分别标注该局部段内的真实缺失区间。
        miss_idx = np.where(local_real_missing[:, j])[0]
        if len(miss_idx) > 0:
            groups = np.split(miss_idx, np.where(np.diff(miss_idx) != 1)[0] + 1)
            for g in groups:
                if len(g) > 0:
                    ax.axhspan(float(d_true[g.min()]), float(d_true[g.max()]), alpha=0.12)

        ax.set_title(name)
        ax.set_xlabel(LOG_UNITS[name])
        ax.grid(True, alpha=0.3)
        apply_log_depth_style(ax, d_true)

    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(format_depth_title("Local uncertainty multi-samples | all logs", csv_path.name, top, bottom))
    plt.tight_layout()
    save_path = output_dir / f"{csv_path.stem}_local_uncertainty_multi_samples.png"
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
    # 默认从测试井中随机抽取案例井；每次运行 Generate Cases 都可能不同。
    rng = random.Random()
    selected = test_files.copy()
    rng.shuffle(selected)
    selected = selected[:num_cases]

    run_dir = make_run_dir("generate_cases_resnet")
    logger.info(f"Random selected cases: {[p.name for p in selected]}")
    logger.info(f"Case outputs will be saved in: {run_dir}")

    # One RNG for this Generate Cases run. It is intentionally not fixed, so every run
    # can choose different windows and different artificial missing intervals.
    case_rng = random.Random()

    for p in selected:
        try:
            # 原始真实缺失案例：整井概览、真实缺失局部放大、多次采样不确定性。
            plot_zoom_and_uncertainty(p, model, scaler, depth_mean, depth_std, run_dir)
        except Exception as e:
            logger.error(f"Generate real-missing case failed: {p.name}: {e}")

        try:
            # 额外输出该井自己的强条件 / 弱条件人工缺失采样对比。
            # 这样输入 3 时，会对 3 口井分别生成 strong 和 weak，而不是只从全体测试井额外挑 1 组。
            generate_condition_strength_cases_for_file(p, model, scaler, depth_mean, depth_std, run_dir, case_rng)
        except Exception as e:
            logger.error(f"Generate strong/weak condition cases failed: {p.name}: {e}")


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
    run_dir = make_run_dir(f"one_well_{Path(csv_path).stem}_resnet")
    logger.info(f"Single-well outputs will be saved in: {run_dir}")
    plot_zoom_and_uncertainty(Path(csv_path), model, scaler, depth_mean, depth_std, run_dir)


def show_model_status():
    path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LATEST_MODEL_PATH
    if not path.exists():
        logger.info("尚未发现模型文件。")
        return
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    logger.info("\n" + "=" * 60)
    logger.info(f"{'WellLog CDDPM ResNet 状态':^50}")
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
    run_dir = make_run_dir("loss_resnet")
    save_path = run_dir / "welllog_cddpm_resnet_loss.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved loss plot: {save_path}")


def main():
    set_seed(SEED)
    logger.info(f"Device: {DEVICE}")
    while True:
        logger.info("\n" + "=" * 55)
        logger.info("        WellLog CDDPM ResNet 管理系统")
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



# ============================================================
# 8.1 Rewritten Generate Cases: one real-missing well, strong/weak windows
# ============================================================
REAL_MISSING_MC_SAMPLES = 3


def _real_missing_intervals_in_window(m_obs_window: np.ndarray, ch: int, min_len: int = 1) -> List[Tuple[int, int]]:
    """Return true-missing intervals for one curve inside a window.

    Args:
        m_obs_window: [C, L], 1=observed, 0=true missing.
        ch: channel index.
        min_len: discard very short missing runs.

    Important:
        For Generate Cases, min_len defaults to 1 because every true-missing
        point inside the selected window should be treated as a target. This
        prevents STRONG figures from leaving other missing points unfilled.
    """
    miss = np.asarray(m_obs_window[ch] < 0.5, dtype=bool)
    intervals = missing_intervals(miss)
    return [(s, e) for s, e in intervals if (e - s) >= min_len]


def _make_target_from_intervals(c: int, l: int, intervals_with_ch: List[Tuple[int, int, int]]) -> np.ndarray:
    """Build target mask [C, L] from selected real-missing intervals.

    intervals_with_ch stores (ch, start, end).
    """
    target = np.zeros((c, l), dtype=np.float32)
    for ch, s, e in intervals_with_ch:
        target[ch, s:e] = 1.0
    return target


def _count_segments_by_channel(intervals_with_ch: List[Tuple[int, int, int]]) -> Dict[int, int]:
    out = {}
    for ch, _, _ in intervals_with_ch:
        out[ch] = out.get(ch, 0) + 1
    return out


def _collect_real_missing_window_candidates(
    csv_path: Path,
    scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    rng: random.Random,
    weak_min_segments: int = 2,
    weak_max_segments: int = 3,
):
    """Find STRONG and WEAK windows from TRUE missing intervals of one well.

    Updated rule according to the requested Generate Cases behavior:

    STRONG:
        The selected window must contain exactly ONE true-missing segment in total
        across all curves. The target mask equals ALL true-missing values in this
        window, so there will be no remaining unfilled missing segment in the plot.

    WEAK:
        The selected window must contain TWO or THREE true-missing segments in total
        across all curves. The target mask also equals ALL true-missing values in
        this window. If possible, weak windows with missing segments on the same
        curve are preferred, because they make the weak-condition contrast clearer.
    """
    df = read_one_csv(csv_path)
    raw_phys = df[LOG_NAMES].values.astype(np.float32)

    # This well must have true missing values, otherwise it is not a valid case well.
    if not np.any(~np.isfinite(raw_phys)):
        return None, None

    x, m, depth = preprocess_well_for_model(df, scaler)
    xw, mw, dw, spans = make_windows(x, m, depth, depth_mean, depth_std)
    if len(xw) == 0:
        return None, None

    strong_candidates = []
    weak_candidates = []
    c = len(LOG_NAMES)
    l = WINDOW_SIZE

    indices = list(range(len(xw)))
    rng.shuffle(indices)

    for idx in indices:
        # Collect ALL real-missing intervals in this window, not just selected ones.
        # This is the key fix: target must cover every true-missing segment inside
        # the selected window, especially for STRONG.
        all_intervals = []
        for ch in range(c):
            intervals = _real_missing_intervals_in_window(mw[idx], ch, min_len=1)
            for s, e in intervals:
                all_intervals.append((ch, s, e))

        if not all_intervals:
            continue

        num_segments = len(all_intervals)
        start, end = spans[idx]
        target = _make_target_from_intervals(c, l, all_intervals)
        sample_dict = {
            "csv_path": csv_path,
            "window_idx": idx,
            "span": (start, end),
            "target_intervals": all_intervals,
            "depth": depth[start:end],
            "x0": torch.from_numpy(xw[idx][None]).float(),
            "m_obs": torch.from_numpy(mw[idx][None]).float(),
            "dw": torch.from_numpy(dw[idx][None]).float(),
            "target": torch.from_numpy(target[None]).float(),
        }

        if num_segments == 1:
            strong_sample = dict(sample_dict)
            strong_sample["case_type"] = "STRONG"
            strong_candidates.append(strong_sample)

        if weak_min_segments <= num_segments <= weak_max_segments:
            weak_sample = dict(sample_dict)
            weak_sample["case_type"] = "WEAK"
            seg_by_ch = _count_segments_by_channel(all_intervals)
            weak_sample["same_curve_segment_max"] = max(seg_by_ch.values()) if seg_by_ch else 0
            weak_candidates.append(weak_sample)

    strong_sample = rng.choice(strong_candidates) if strong_candidates else None

    if weak_candidates:
        # Prefer weak windows where two or three missing segments are on the same curve.
        preferred = [s for s in weak_candidates if s.get("same_curve_segment_max", 0) >= weak_min_segments]
        weak_sample = rng.choice(preferred if preferred else weak_candidates)
        weak_sample.pop("same_curve_segment_max", None)
    else:
        weak_sample = None

    return strong_sample, weak_sample


@torch.no_grad()
def plot_real_missing_sampling_case(
    sample: Dict,
    model: CDDPM,
    scaler: SimpleStandardScaler,
    save_path: Path,
    num_samples: int = REAL_MISSING_MC_SAMPLES,
):
    """Plot one real-missing window with multiple imputation samples.

    No condition curve is drawn.
    The figure contains:
        - observed/original curve: only originally observed values
        - sample 1~N: model imputations, only target real-missing segments are filled
        - shaded target intervals
    """
    x0 = sample["x0"].to(model.device)
    m_obs = sample["m_obs"].to(model.device)
    target = sample["target"].to(model.device)
    depth_t = sample["dw"].to(model.device)
    true_depth = sample["depth"].astype(np.float32)
    target_np = target[0].detach().cpu().numpy()
    m_obs_np = m_obs[0].detach().cpu().numpy()

    observed = model_to_physical_values(x0[0].detach().cpu().numpy().T, scaler)
    # Hide true-missing positions in the observed/original curve.
    observed[m_obs_np.T < 0.5] = np.nan

    sample_phys_list = []
    for k in range(num_samples):
        # For real missing, x0 already has filled zeros at missing positions.
        # impute() uses m_cond = m_obs * (1 - target), so only observed data are conditions.
        x_hat = model.impute(x0, m_obs, depth_t, target_mask=target)
        pred_phys = model_to_physical_values(x_hat[0].detach().cpu().numpy().T, scaler)

        # Only show model-generated values in the selected target real-missing intervals.
        # Other non-target true-missing values stay NaN to avoid drawing fake mean-filled curves.
        display_curve = observed.copy()
        target_bool = target_np.T > 0.5
        display_curve[target_bool] = pred_phys[target_bool]
        sample_phys_list.append(display_curve)
        logger.info(f"  {sample['case_type']} real-missing sample {k + 1}/{num_samples} done")

    samples = np.stack(sample_phys_list, axis=0)  # [S, L, C]

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(16, 7), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        ax.plot(observed[:, j], true_depth, color="black", linewidth=1.8, label="observed")

        for k in range(num_samples):
            ax.plot(
                samples[k, :, j],
                true_depth,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label=f"sample {k + 1}" if j == len(LOG_NAMES) - 1 else None,
            )

        # Shade selected target intervals only.
        idx = np.where(target_np[j] > 0.5)[0]
        if len(idx) > 0:
            groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for g in groups:
                if len(g) > 0:
                    ax.axhspan(float(true_depth[g.min()]), float(true_depth[g.max()]), alpha=0.15)

        ax.set_title(name)
        ax.set_xlabel(LOG_UNITS[name])
        apply_log_depth_style(ax, true_depth)

    axes[0].set_ylabel(f"Depth ({LOG_UNITS['Depth']})")
    axes[-1].legend(fontsize=8, loc="best")

    top, bottom = depth_range(true_depth)
    interval_desc = []
    for ch, s, e in sample["target_intervals"]:
        interval_desc.append(f"{LOG_NAMES[ch]}[{s}:{e}]")
    interval_text = ", ".join(interval_desc)

    fig.suptitle(
        format_depth_title(
            f"{sample['case_type']} real-missing imputation | {num_samples} samples | target: {interval_text}",
            sample["csv_path"].name,
            top,
            bottom,
        )
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    logger.info(f"Saved real-missing sampling case: {save_path.name}")


def generate_cases():
    """[3] Generate Cases: rewritten to match the requested workflow.

    New behavior:
        1. Generate cases for only ONE well each time.
        2. The well must contain true missing values.
        3. Randomly select two windows from that well:
           - STRONG: one true-missing segment to impute.
           - WEAK: two or three true-missing segments to impute.
        4. For each window, sample 3 times.
        5. Do not draw condition curves.
    """
    _, test_files = load_split_files()
    model, scaler, depth_mean, depth_std, _ = load_model(prefer_best=True)

    rng = random.Random()
    candidate_files = test_files.copy()
    rng.shuffle(candidate_files)

    run_dir = make_run_dir("generate_one_real_missing_well_strong_weak")
    logger.info(f"Case outputs will be saved in: {run_dir}")

    chosen_file = None
    chosen_strong = None
    chosen_weak = None

    for p in candidate_files:
        try:
            strong_sample, weak_sample = _collect_real_missing_window_candidates(
                p, scaler, depth_mean, depth_std, rng
            )
            if strong_sample is not None and weak_sample is not None:
                chosen_file = p
                chosen_strong = strong_sample
                chosen_weak = weak_sample
                break
            else:
                logger.info(
                    f"[CASE-SKIP] {p.name}: cannot find both STRONG and WEAK real-missing windows "
                    f"(strong={strong_sample is not None}, weak={weak_sample is not None})"
                )
        except Exception as e:
            logger.info(f"[CASE-SKIP] {p.name}: {e}")

    if chosen_file is None:
        logger.warning("No test well found with both STRONG and WEAK real-missing windows.")
        logger.warning("Suggestion: check whether test wells contain true missing intervals after CSV cleaning.")
        return

    logger.info(f"Selected real-missing case well: {chosen_file.name}")

    s0, s1 = chosen_strong["span"]
    strong_path = run_dir / (
        f"real_missing_STRONG_{chosen_file.stem}_win{chosen_strong['window_idx']}_"
        f"idx{s0}_{s1}_mc{REAL_MISSING_MC_SAMPLES}.png"
    )
    plot_real_missing_sampling_case(
        chosen_strong,
        model,
        scaler,
        strong_path,
        num_samples=REAL_MISSING_MC_SAMPLES,
    )

    w0, w1 = chosen_weak["span"]
    weak_path = run_dir / (
        f"real_missing_WEAK_{chosen_file.stem}_win{chosen_weak['window_idx']}_"
        f"idx{w0}_{w1}_mc{REAL_MISSING_MC_SAMPLES}.png"
    )
    plot_real_missing_sampling_case(
        chosen_weak,
        model,
        scaler,
        weak_path,
        num_samples=REAL_MISSING_MC_SAMPLES,
    )

    logger.info("Generate Cases finished.")
    logger.info(f"Selected well: {chosen_file.name}")
    logger.info(f"STRONG figure: {strong_path}")
    logger.info(f"WEAK figure: {weak_path}")

if __name__ == "__main__":
    main()
