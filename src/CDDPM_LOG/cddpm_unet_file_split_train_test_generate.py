"""
CDDPM for multi-well CSV well-log imputation with 1D Conditional U-Net backbone.

CSV columns required:
    Depth, GR, RHOB, RILD, CNPOR

Main features:
    1. Read multiple CSV files, one file = one well.
    2. Preserve real missing mask m_obs.
    3. Make sliding windows inside each well only.
    4. Use two depth condition channels:
        - relative depth in window: 0~1
        - standardized absolute depth
    5. Train CDDPM by self-supervised artificial masking.
    6. Evaluate artificial-missing reconstruction on test windows.
    7. Impute real missing values and export *_imputed.csv.

Author: ChatGPT
"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional: reduce torch dynamo initialization overhead on some PyTorch versions.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

# ============================================================
# 1. Config
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_DIR = "/home/tet/zhaoheng/fast-api-project/data/2023_log_csv"
OUTPUT_DIR = "/home/tet/zhaoheng/fast-api-project/src/CDDPM_LOG/outputs_unet_cddpm"

LOG_NAMES = ["GR", "RHOB", "RILD", "CNPOR"]
REQUIRED_COLUMNS = ["Depth"] + LOG_NAMES

WINDOW_SIZE = 128
WINDOW_STRIDE = 32
MIN_OBS_RATIO = 0.50

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4

DIFFUSION_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

TARGET_MIN_LEN = 16
TARGET_MAX_LEN = 48
MIN_TARGET_CHANNELS = 1
MAX_TARGET_CHANNELS = 2

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0

# Default mode. You can override it from command line: --mode train/test/generate/all
MODE = "train"

# U-Net settings
BASE_CHANNELS = 64
TIME_DIM = 128
DEPTH_CHANNELS = 16

# Evaluation / sampling
EVAL_BATCHES = 4
MC_SAMPLES = 1  # set >1 to estimate uncertainty, but slower

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(Path(OUTPUT_DIR) / "cddpm_unet_model.pt")
FULL_CHECKPOINT_PATH = str(Path(OUTPUT_DIR) / "cddpm_unet_full_checkpoint.pt")
LOAD_CHECKPOINT_PATH = FULL_CHECKPOINT_PATH
SPLIT_FILE_PATH = str(Path(OUTPUT_DIR) / "well_file_split.npz")


# ============================================================
# 2. Utils
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 3. Dataset
# ============================================================
class WellLogCDDPMDataset(Dataset):
    def __init__(
        self,
        x_windows: np.ndarray,
        obs_masks: np.ndarray,
        depth_windows: np.ndarray,
        target_min_len: int = 16,
        target_max_len: int = 48,
        min_target_channels: int = 1,
        max_target_channels: int = 2,
    ):
        self.x = torch.from_numpy(x_windows).float()
        self.m_obs = torch.from_numpy(obs_masks).float()
        self.depth = torch.from_numpy(depth_windows).float()
        self.target_min_len = target_min_len
        self.target_max_len = target_max_len
        self.min_target_channels = min_target_channels
        self.max_target_channels = max_target_channels

    def __len__(self):
        return len(self.x)

    def _make_target_mask(self, x0: torch.Tensor, m_obs: torch.Tensor) -> torch.Tensor:
        c, l = x0.shape
        m_ta = torch.zeros_like(x0)
        num_ch = random.randint(self.min_target_channels, min(self.max_target_channels, c))
        channels = random.sample(range(c), k=num_ch)

        for ch in channels:
            seg_len = random.randint(self.target_min_len, self.target_max_len)
            seg_len = min(seg_len, max(1, l // 2))
            found = False

            for _ in range(50):
                start = random.randint(0, l - seg_len)
                end = start + seg_len
                if m_obs[ch, start:end].mean() > 0.95:
                    m_ta[ch, start:end] = 1.0
                    found = True
                    break

            if not found:
                valid_idx = torch.where(m_obs[ch] > 0.5)[0]
                if len(valid_idx) > seg_len:
                    start_pos = random.randint(0, len(valid_idx) - seg_len)
                    chosen = valid_idx[start_pos:start_pos + seg_len]
                    m_ta[ch, chosen] = 1.0

        return m_ta

    def __getitem__(self, idx):
        x0 = self.x[idx].clone()
        m_obs = self.m_obs[idx].clone()
        depth = self.depth[idx].clone()

        m_ta = self._make_target_mask(x0, m_obs)
        m_cond = m_obs * (1.0 - m_ta)
        x_cond = x0 * m_cond

        return {
            "x0": x0,
            "m_obs": m_obs,
            "m_ta": m_ta,
            "m_cond": m_cond,
            "x_cond": x_cond,
            "depth": depth,
        }


class SimpleStandardScaler:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "SimpleStandardScaler":
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


def find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    col_map = {c.upper(): c for c in df.columns}
    return col_map.get(target.upper())


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame()
    for col in REQUIRED_COLUMNS:
        matched = find_col(df, col)
        if matched is None:
            raise ValueError(f"missing required column {col}")
        out[col] = pd.to_numeric(df[matched], errors="coerce")

    out = out.dropna(subset=["Depth"])
    out = out.sort_values("Depth").reset_index(drop=True)

    # Basic physical cleaning. You can adjust these ranges.
    out["GR"] = out["GR"].mask(out["GR"] < 0)
    out["RHOB"] = out["RHOB"].mask((out["RHOB"] < 1.0) | (out["RHOB"] > 4.0))
    out["RILD"] = out["RILD"].mask(out["RILD"] <= 0)

    # CNPOR sometimes is in percent, e.g. 25, convert to fraction.
    if out["CNPOR"].dropna().shape[0] > 0 and out["CNPOR"].dropna().median() > 1.5:
        out["CNPOR"] = out["CNPOR"] / 100.0
    out["CNPOR"] = out["CNPOR"].mask((out["CNPOR"] < -0.2) | (out["CNPOR"] > 1.0))

    # Log-transform resistivity after invalid values are masked.
    out["RILD"] = np.log10(out["RILD"])
    return out


def scan_csv_files(csv_dir: str) -> List[Path]:
    paths = sorted(list(Path(csv_dir).glob("*.csv")) + list(Path(csv_dir).glob("*.CSV")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No csv files found in {csv_dir}")
    return paths


def collect_scaler_stats(csv_paths: List[Path]) -> Tuple[SimpleStandardScaler, float, float, List[Tuple[Path, pd.DataFrame]]]:
    """Read valid wells and fit scaler on the provided wells only.

    Important: in train mode this function is called with train wells only,
    so validation/test wells do not leak into normalization statistics.
    """
    valid = []
    all_logs = []
    all_depth = []

    for p in csv_paths:
        try:
            df = read_one_csv(p)
            if len(df) < WINDOW_SIZE:
                print(f"[SKIP-short] {p.name}: length={len(df)}")
                continue
            values = df[LOG_NAMES].values.astype(np.float32)
            depth = df["Depth"].values.astype(np.float32)
            if np.isfinite(values).mean() < MIN_OBS_RATIO:
                print(f"[SKIP-lowobs] {p.name}: obs={np.isfinite(values).mean():.3f}")
                continue
            all_logs.append(values)
            all_depth.append(depth.reshape(-1, 1))
            valid.append((p, df))
        except Exception as e:
            print(f"[SKIP-error] {p.name}: {e}")

    if len(valid) == 0:
        raise RuntimeError("No valid wells after scanning CSV files.")

    all_logs_arr = np.concatenate(all_logs, axis=0)
    log_scaler = SimpleStandardScaler().fit(all_logs_arr)

    all_depth_arr = np.concatenate(all_depth, axis=0)
    depth_mean = float(np.nanmean(all_depth_arr))
    depth_std = float(np.nanstd(all_depth_arr))
    if depth_std < 1e-6:
        depth_std = 1.0

    print(f"Valid wells for scaler: {len(valid)} / {len(csv_paths)}")
    print(f"Observed log ratio for scaler: {np.isfinite(all_logs_arr).mean():.4f}")
    print(f"Depth mean/std: {depth_mean:.3f} / {depth_std:.3f}")
    return log_scaler, depth_mean, depth_std, valid


def split_csv_files_by_well(
    csv_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
    split_file: str = SPLIT_FILE_PATH,
    use_cache: bool = True,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split CSV files by well/file, not by windows.

    The split is cached so that train/test mode use exactly the same wells.
    """
    csv_paths = scan_csv_files(csv_dir)
    path_map = {p.name: p for p in csv_paths}

    if use_cache and Path(split_file).exists():
        data = np.load(split_file, allow_pickle=True)
        train_names = [str(x) for x in data["train"]]
        val_names = [str(x) for x in data["val"]]
        test_names = [str(x) for x in data["test"]]
        train_paths = [path_map[n] for n in train_names if n in path_map]
        val_paths = [path_map[n] for n in val_names if n in path_map]
        test_paths = [path_map[n] for n in test_names if n in path_map]
        print(f"Loaded cached file split: {split_file}")
        print(f"Split wells: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
        return train_paths, val_paths, test_paths

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(csv_paths))
    rng.shuffle(idx)
    shuffled = [csv_paths[i] for i in idx]

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:n_train + n_val]
    test_paths = shuffled[n_train + n_val:]

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_file,
        train=np.array([p.name for p in train_paths], dtype=object),
        val=np.array([p.name for p in val_paths], dtype=object),
        test=np.array([p.name for p in test_paths], dtype=object),
    )
    print(f"Saved file split to: {split_file}")
    print(f"Split wells: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    return train_paths, val_paths, test_paths


def preprocess_one_well(df: pd.DataFrame, log_scaler: SimpleStandardScaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[LOG_NAMES].values.astype(np.float32)
    depth = df["Depth"].values.astype(np.float32)
    obs_mask = np.isfinite(values).astype(np.float32)
    x_scaled = log_scaler.transform(values).astype(np.float32)
    x_filled = np.where(np.isfinite(x_scaled), x_scaled, 0.0).astype(np.float32)
    return x_filled, obs_mask, depth


def make_windows_one_well(
    x: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    well_name: str,
    depth_mean: float,
    depth_std: float,
    window_size: int,
    stride: int,
    min_obs_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    x: [N, C]
    mask: [N, C]
    depth: [N]

    Returns:
        xw: [num_windows, C, L]
        mw: [num_windows, C, L]
        dw: [num_windows, 2, L]
        info: per-window metadata
    """
    x_windows, m_windows, d_windows, info = [], [], [], []
    n = len(x)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        xw = x[start:end]
        mw = mask[start:end]
        dw_abs = depth[start:end]

        if mw.mean() < min_obs_ratio:
            continue

        rel_depth = np.linspace(0.0, 1.0, window_size, dtype=np.float32)
        abs_depth_scaled = ((dw_abs - depth_mean) / depth_std).astype(np.float32)
        depth_feature = np.stack([rel_depth, abs_depth_scaled], axis=0).astype(np.float32)

        x_windows.append(xw)
        m_windows.append(mw)
        d_windows.append(depth_feature)
        info.append({
            "well_name": well_name,
            "start_idx": start,
            "end_idx": end,
            "start_depth": float(dw_abs[0]),
            "end_depth": float(dw_abs[-1]),
        })

    if len(x_windows) == 0:
        return (
            np.empty((0, len(LOG_NAMES), window_size), dtype=np.float32),
            np.empty((0, len(LOG_NAMES), window_size), dtype=np.float32),
            np.empty((0, 2, window_size), dtype=np.float32),
            [],
        )

    x_windows = np.stack(x_windows, axis=0)  # [Nw, L, C]
    m_windows = np.stack(m_windows, axis=0)  # [Nw, L, C]
    d_windows = np.stack(d_windows, axis=0)  # [Nw, 2, L]

    x_windows = np.transpose(x_windows, (0, 2, 1))
    m_windows = np.transpose(m_windows, (0, 2, 1))
    return x_windows.astype(np.float32), m_windows.astype(np.float32), d_windows.astype(np.float32), info


def build_windows_from_files(
    csv_paths: List[Path],
    log_scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    split_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], List[Tuple[Path, pd.DataFrame]]]:
    """Build windows only from the specified well files."""
    all_x, all_m, all_d, all_info = [], [], [], []
    valid = []

    for p in csv_paths:
        try:
            df = read_one_csv(p)
            if len(df) < WINDOW_SIZE:
                print(f"[SKIP-short-{split_name}] {p.name}: length={len(df)}")
                continue
            values = df[LOG_NAMES].values.astype(np.float32)
            if np.isfinite(values).mean() < MIN_OBS_RATIO:
                print(f"[SKIP-lowobs-{split_name}] {p.name}: obs={np.isfinite(values).mean():.3f}")
                continue

            x, m, depth = preprocess_one_well(df, log_scaler)
            xw, mw, dw, info = make_windows_one_well(
                x=x,
                mask=m,
                depth=depth,
                well_name=p.stem,
                depth_mean=depth_mean,
                depth_std=depth_std,
                window_size=WINDOW_SIZE,
                stride=WINDOW_STRIDE,
                min_obs_ratio=MIN_OBS_RATIO,
            )
            if len(xw) > 0:
                all_x.append(xw)
                all_m.append(mw)
                all_d.append(dw)
                all_info.extend(info)
                valid.append((p, df))
            print(f"[WINDOW-{split_name}] {p.name}: {len(xw)}")
        except Exception as e:
            print(f"[SKIP-error-{split_name}] {p.name}: {e}")

    if len(all_x) == 0:
        raise RuntimeError(f"No windows generated for split: {split_name}")

    all_x = np.concatenate(all_x, axis=0)
    all_m = np.concatenate(all_m, axis=0)
    all_d = np.concatenate(all_d, axis=0)

    print(f"{split_name} window x shape:", all_x.shape)
    print(f"{split_name} window mask shape:", all_m.shape)
    print(f"{split_name} window depth shape:", all_d.shape)

    return all_x, all_m, all_d, all_info, valid


def make_dataset_and_loader(
    x: np.ndarray,
    m: np.ndarray,
    d: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> Tuple[WellLogCDDPMDataset, DataLoader]:
    dataset = WellLogCDDPMDataset(
        x,
        m,
        d,
        target_min_len=TARGET_MIN_LEN,
        target_max_len=TARGET_MAX_LEN,
        min_target_channels=MIN_TARGET_CHANNELS,
        max_target_channels=MAX_TARGET_CHANNELS,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    return dataset, loader




# ============================================================
# 4. Embeddings and 1D Conditional U-Net
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


def valid_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class UNetResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = valid_group_norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = valid_group_norm(out_ch)
        self.act = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class DownBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res1 = UNetResBlock1D(in_ch, out_ch, time_dim)
        self.res2 = UNetResBlock1D(out_ch, out_ch, time_dim)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res1 = UNetResBlock1D(out_ch + skip_ch, out_ch, time_dim)
        self.res2 = UNetResBlock1D(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        # Handle odd sequence lengths if necessary.
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class ConditionalUNet1D(nn.Module):
    """
    1D Conditional U-Net denoiser.

    Inputs:
        x_ta_t: [B, C, L] noisy target region, non-target region is zero
        x_cond: [B, C, L] condition logs, hidden region is zero
        m_cond: [B, C, L] condition mask, 1=visible, 0=invisible
        depth:  [B, 2, L] relative depth + standardized absolute depth
        t:      [B]

    Output:
        pred_noise: [B, C, L]
    """
    def __init__(
        self,
        log_channels: int = 4,
        depth_in_channels: int = 2,
        base_channels: int = 64,
        time_dim: int = 128,
        depth_channels: int = 16,
    ):
        super().__init__()
        self.log_channels = log_channels

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.depth_proj = nn.Sequential(
            nn.Conv1d(depth_in_channels, depth_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(depth_channels, depth_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        in_ch = log_channels * 3 + depth_channels
        self.init_conv = nn.Conv1d(in_ch, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock1D(base_channels, base_channels, time_dim)
        self.down2 = DownBlock1D(base_channels, base_channels * 2, time_dim)
        self.down3 = DownBlock1D(base_channels * 2, base_channels * 4, time_dim)

        self.mid1 = UNetResBlock1D(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = UNetResBlock1D(base_channels * 4, base_channels * 4, time_dim)

        self.up3 = UpBlock1D(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up2 = UpBlock1D(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up1 = UpBlock1D(base_channels, base_channels, base_channels, time_dim)

        self.out = nn.Sequential(
            valid_group_norm(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, log_channels, kernel_size=1),
        )

    def forward(self, x_ta_t, x_cond, m_cond, depth, t):
        t_emb = self.time_emb(t)
        d_emb = self.depth_proj(depth)
        x = torch.cat([x_ta_t, x_cond, m_cond, d_emb], dim=1)

        x = self.init_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out(x)


# ============================================================
# 5. CDDPM
# ============================================================
class CDDPM:
    def __init__(self, model: nn.Module, timesteps: int, beta_start: float, beta_end: float, device: str):
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

        denom = m_ta.sum().clamp(min=1.0)
        loss = (((pred_noise - noise) ** 2) * m_ta).sum() / denom
        return loss

    @torch.no_grad()
    def impute(self, x0_filled: torch.Tensor, m_obs: torch.Tensor, depth: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        self.model.eval()
        x0_filled = x0_filled.to(self.device)
        m_obs = m_obs.to(self.device)
        depth = depth.to(self.device)

        if target_mask is None:
            m_target = 1.0 - m_obs
        else:
            m_target = target_mask.to(self.device)

        m_cond = 1.0 - m_target
        x_cond = x0_filled * m_cond
        x = torch.randn_like(x0_filled) * m_target

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, x_cond, m_cond, depth, t)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)

            if step > 0:
                noise = torch.randn_like(x)
                x_prev = mean + torch.sqrt(beta_t) * noise
            else:
                x_prev = mean

            x = x_prev * m_target

        x_completed = x_cond + x * m_target
        return x_completed


# ============================================================
# 6. Training and evaluation
# ============================================================
def train(cddpm: CDDPM, train_loader: DataLoader, test_loader: DataLoader, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(cddpm.model.parameters(), lr=lr)
    best_test = float("inf")

    for epoch in range(1, epochs + 1):
        cddpm.model.train()
        total_loss = 0.0
        count = 0

        for batch in train_loader:
            loss = cddpm.p_losses(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cddpm.model.parameters(), 1.0)
            optimizer.step()

            bs = batch["x0"].size(0)
            total_loss += loss.item() * bs
            count += bs

        train_loss = total_loss / max(count, 1)
        test_loss = estimate_loss(cddpm, test_loader, max_batches=10)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | test_noise_loss={test_loss:.6f}")

        if test_loss < best_test:
            best_test = test_loss
            torch.save({"model": cddpm.model.state_dict()}, MODEL_PATH)
            print(f"  saved best model: {MODEL_PATH}")


@torch.no_grad()
def estimate_loss(cddpm: CDDPM, loader: DataLoader, max_batches: int = 10) -> float:
    cddpm.model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss = cddpm.p_losses(batch)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_imputation(cddpm: CDDPM, loader: DataLoader, max_batches: int = 4) -> Dict[str, float]:
    """Evaluate artificial target imputation on standardized data."""
    cddpm.model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    n_sum = 0.0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x0 = batch["x0"].to(cddpm.device)
        m_obs = batch["m_obs"].to(cddpm.device)
        m_ta = batch["m_ta"].to(cddpm.device)
        depth = batch["depth"].to(cddpm.device)

        # For artificial test, hide target region, but other real observed points remain condition.
        x_masked = x0 * (1.0 - m_ta)
        m_cond_for_eval = m_obs * (1.0 - m_ta)

        x_hat = cddpm.impute(x_masked, m_cond_for_eval, depth, target_mask=m_ta)
        diff = (x_hat - x0) * m_ta
        se_sum += float((diff ** 2).sum().item())
        ae_sum += float(diff.abs().sum().item())
        n_sum += float(m_ta.sum().item())

    mse = se_sum / max(n_sum, 1.0)
    rmse = math.sqrt(mse)
    mae = ae_sum / max(n_sum, 1.0)
    return {"MAE_std": mae, "RMSE_std": rmse, "MSE_std": mse}


def plot_test_example(cddpm: CDDPM, test_loader: DataLoader, save_path: str) -> None:
    batch = next(iter(test_loader))
    x0 = batch["x0"][:1].to(cddpm.device)
    m_obs = batch["m_obs"][:1].to(cddpm.device)
    m_ta = batch["m_ta"][:1].to(cddpm.device)
    depth = batch["depth"][:1].to(cddpm.device)

    x_masked = x0 * (1.0 - m_ta)
    m_cond = m_obs * (1.0 - m_ta)
    x_hat = cddpm.impute(x_masked, m_cond, depth, target_mask=m_ta)

    original = x0[0].detach().cpu().numpy()
    masked = x_masked[0].detach().cpu().numpy()
    imputed = x_hat[0].detach().cpu().numpy()
    target = m_ta[0].detach().cpu().numpy()
    depth_rel = depth[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(14, 6), sharey=True)
    for i, name in enumerate(LOG_NAMES):
        ax = axes[i]
        ax.plot(original[i], depth_rel, label="original")
        ax.plot(masked[i], depth_rel, label="condition")
        ax.plot(imputed[i], depth_rel, linestyle="--", label="imputed")
        idx = np.where(target[i] > 0.5)[0]
        if len(idx) > 0:
            ax.axhspan(depth_rel[idx.min()], depth_rel[idx.max()], alpha=0.15)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("relative depth")
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved test example: {save_path}")


# ============================================================
# 7. Real missing imputation and export
# ============================================================
@torch.no_grad()
def impute_one_well_export(
    cddpm: CDDPM,
    csv_path: Path,
    log_scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    output_dir: str,
) -> None:
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_one_well(df, log_scaler)

    # If no real missing values, simply export a copy with normalized cleaning applied.
    if (m < 0.5).sum() == 0:
        out = df.copy()
        out_path = Path(output_dir) / f"{csv_path.stem}_imputed.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    # Window-based imputation with overlap averaging.
    n, c = x.shape
    pred_sum = np.zeros_like(x, dtype=np.float32)
    pred_count = np.zeros_like(x, dtype=np.float32)

    starts = list(range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE)) if n >= WINDOW_SIZE else []
    if len(starts) == 0 or starts[-1] != n - WINDOW_SIZE:
        if n >= WINDOW_SIZE:
            starts.append(n - WINDOW_SIZE)

    for start in starts:
        end = start + WINDOW_SIZE
        xw = x[start:end]
        mw = m[start:end]
        dw_abs = depth[start:end]

        rel_depth = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
        abs_depth_scaled = ((dw_abs - depth_mean) / depth_std).astype(np.float32)
        dw = np.stack([rel_depth, abs_depth_scaled], axis=0).astype(np.float32)

        xw_t = torch.from_numpy(np.transpose(xw, (1, 0))[None]).float().to(cddpm.device)
        mw_t = torch.from_numpy(np.transpose(mw, (1, 0))[None]).float().to(cddpm.device)
        dw_t = torch.from_numpy(dw[None]).float().to(cddpm.device)

        completed = cddpm.impute(xw_t, mw_t, dw_t, target_mask=None)[0].detach().cpu().numpy()  # [C,L]
        completed = np.transpose(completed, (1, 0))  # [L,C]

        missing = 1.0 - mw
        pred_sum[start:end] += completed * missing
        pred_count[start:end] += missing

    x_completed = x.copy()
    fill_mask = pred_count > 0
    x_completed[fill_mask] = pred_sum[fill_mask] / np.maximum(pred_count[fill_mask], 1e-6)

    logs_completed = log_scaler.inverse_transform(x_completed)

    out = df.copy()
    for j, name in enumerate(LOG_NAMES):
        raw = df[name].values.astype(np.float32)
        is_missing = ~np.isfinite(raw)
        out[f"{name}_original"] = raw
        out[f"{name}_imputed"] = logs_completed[:, j]
        out[name] = np.where(is_missing, logs_completed[:, j], raw)

    out_path = Path(output_dir) / f"{csv_path.stem}_imputed.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[IMPUTED] {csv_path.name} -> {out_path.name}")


# ============================================================
# 8. Main
# ============================================================
def build_cddpm_from_checkpoint_config(checkpoint: Optional[Dict] = None) -> CDDPM:
    if checkpoint is not None and "config" in checkpoint:
        cfg = checkpoint["config"]
        base_channels = cfg.get("base_channels", BASE_CHANNELS)
        time_dim = cfg.get("time_dim", TIME_DIM)
        depth_channels = cfg.get("depth_channels", DEPTH_CHANNELS)
    else:
        base_channels = BASE_CHANNELS
        time_dim = TIME_DIM
        depth_channels = DEPTH_CHANNELS

    log_names = checkpoint.get("log_names", LOG_NAMES) if checkpoint is not None else LOG_NAMES
    diffusion_steps = checkpoint.get("diffusion_steps", DIFFUSION_STEPS) if checkpoint is not None else DIFFUSION_STEPS

    model = ConditionalUNet1D(
        log_channels=len(log_names),
        depth_in_channels=2,
        base_channels=base_channels,
        time_dim=time_dim,
        depth_channels=depth_channels,
    )
    return CDDPM(
        model=model,
        timesteps=diffusion_steps,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=DEVICE,
    )



def load_trained_cddpm(checkpoint_path: str):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    log_scaler = SimpleStandardScaler()
    log_scaler.load_state_dict(checkpoint["log_scaler"])
    depth_mean = float(checkpoint["depth_mean"])
    depth_std = float(checkpoint["depth_std"])

    cddpm = build_cddpm_from_checkpoint_config(checkpoint)
    cddpm.model.load_state_dict(checkpoint["model"])
    cddpm.model.eval()
    print("Model loaded.")
    return cddpm, log_scaler, depth_mean, depth_std, checkpoint


def select_case_files(
    test_files: List[Path],
    all_files: List[Path],
    case_wells: Optional[str],
    num_cases: int,
) -> List[Path]:
    """Select 1~3 case wells for final visualization.

    If --case-wells is provided, names can be either csv file names or stems.
    Otherwise, use the first num_cases files from the held-out test split.
    """
    if case_wells:
        name_map = {}
        for p in all_files:
            name_map[p.name] = p
            name_map[p.stem] = p
        selected = []
        for item in case_wells.split(','):
            key = item.strip()
            if not key:
                continue
            if key in name_map:
                selected.append(name_map[key])
            elif key + ".csv" in name_map:
                selected.append(name_map[key + ".csv"])
            else:
                print(f"[WARN] requested case well not found: {key}")
        return selected[:num_cases]

    return test_files[:num_cases]


@torch.no_grad()
def plot_artificial_mask_case_for_well(
    cddpm: CDDPM,
    csv_path: Path,
    log_scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    save_path: str,
) -> None:
    """Create one artificial-missing comparison figure for a single well.

    This has ground truth, so it is useful for group-meeting demonstration:
    original vs masked condition vs CDDPM imputed.
    """
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_one_well(df, log_scaler)
    xw, mw, dw, info = make_windows_one_well(
        x=x,
        mask=m,
        depth=depth,
        well_name=csv_path.stem,
        depth_mean=depth_mean,
        depth_std=depth_std,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        min_obs_ratio=MIN_OBS_RATIO,
    )
    if len(xw) == 0:
        print(f"[CASE-SKIP] no valid windows for {csv_path.name}")
        return

    ds = WellLogCDDPMDataset(
        xw, mw, dw,
        target_min_len=TARGET_MIN_LEN,
        target_max_len=TARGET_MAX_LEN,
        min_target_channels=MIN_TARGET_CHANNELS,
        max_target_channels=MAX_TARGET_CHANNELS,
    )

    # Try several windows until an artificial target is successfully produced.
    sample = None
    for i in range(min(len(ds), 30)):
        candidate = ds[i]
        if candidate["m_ta"].sum() > 0:
            sample = candidate
            break
    if sample is None:
        print(f"[CASE-SKIP] cannot create artificial mask for {csv_path.name}")
        return

    x0 = sample["x0"][None].to(cddpm.device)
    m_obs = sample["m_obs"][None].to(cddpm.device)
    m_ta = sample["m_ta"][None].to(cddpm.device)
    depth_t = sample["depth"][None].to(cddpm.device)

    x_masked = x0 * (1.0 - m_ta)
    m_cond = m_obs * (1.0 - m_ta)
    x_hat = cddpm.impute(x_masked, m_cond, depth_t, target_mask=m_ta)

    original_std = x0[0].detach().cpu().numpy().T
    masked_std = x_masked[0].detach().cpu().numpy().T
    imputed_std = x_hat[0].detach().cpu().numpy().T

    original = log_scaler.inverse_transform(original_std).T
    masked = log_scaler.inverse_transform(masked_std).T
    imputed = log_scaler.inverse_transform(imputed_std).T

    target = m_ta[0].detach().cpu().numpy()
    depth_abs_scaled = depth_t[0, 1].detach().cpu().numpy()
    depth_abs = depth_abs_scaled * depth_std + depth_mean

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 7), sharey=True)
    for i, name in enumerate(LOG_NAMES):
        ax = axes[i]
        ax.plot(original[i], depth_abs, label="original")
        ax.plot(masked[i], depth_abs, label="masked condition")
        ax.plot(imputed[i], depth_abs, linestyle="--", label="imputed")
        idx = np.where(target[i] > 0.5)[0]
        if len(idx) > 0:
            ax.axhspan(depth_abs[idx.min()], depth_abs[idx.max()], alpha=0.15)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[-1].legend()
    fig.suptitle(f"Artificial-missing case: {csv_path.name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[CASE] saved artificial-mask figure: {save_path}")


@torch.no_grad()
def plot_real_missing_case_for_well(
    cddpm: CDDPM,
    csv_path: Path,
    log_scaler: SimpleStandardScaler,
    depth_mean: float,
    depth_std: float,
    save_path: str,
) -> None:
    """Plot real-missing imputation for one well.

    There is no ground truth here. The plot shows original available logs and the
    final imputed curve. Shaded intervals indicate real missing points in each log.
    """
    df = read_one_csv(csv_path)
    x, m, depth = preprocess_one_well(df, log_scaler)
    n, c = x.shape

    if (m < 0.5).sum() == 0:
        print(f"[REAL-CASE] {csv_path.name} has no real missing values after cleaning; plotting original only.")

    pred_sum = np.zeros_like(x, dtype=np.float32)
    pred_count = np.zeros_like(x, dtype=np.float32)

    starts = list(range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE)) if n >= WINDOW_SIZE else []
    if n >= WINDOW_SIZE and (len(starts) == 0 or starts[-1] != n - WINDOW_SIZE):
        starts.append(n - WINDOW_SIZE)
    if len(starts) == 0:
        print(f"[REAL-CASE-SKIP] {csv_path.name} shorter than WINDOW_SIZE")
        return

    for start in starts:
        end = start + WINDOW_SIZE
        xw = x[start:end]
        mw = m[start:end]
        dw_abs = depth[start:end]

        rel_depth = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32)
        abs_depth_scaled = ((dw_abs - depth_mean) / depth_std).astype(np.float32)
        dw = np.stack([rel_depth, abs_depth_scaled], axis=0).astype(np.float32)

        xw_t = torch.from_numpy(np.transpose(xw, (1, 0))[None]).float().to(cddpm.device)
        mw_t = torch.from_numpy(np.transpose(mw, (1, 0))[None]).float().to(cddpm.device)
        dw_t = torch.from_numpy(dw[None]).float().to(cddpm.device)

        completed = cddpm.impute(xw_t, mw_t, dw_t, target_mask=None)[0].detach().cpu().numpy()
        completed = np.transpose(completed, (1, 0))
        missing = 1.0 - mw
        pred_sum[start:end] += completed * missing
        pred_count[start:end] += missing

    x_completed = x.copy()
    fill_mask = pred_count > 0
    x_completed[fill_mask] = pred_sum[fill_mask] / np.maximum(pred_count[fill_mask], 1e-6)

    logs_original = log_scaler.inverse_transform(x)
    logs_completed = log_scaler.inverse_transform(x_completed)

    fig, axes = plt.subplots(1, len(LOG_NAMES), figsize=(15, 8), sharey=True)
    for j, name in enumerate(LOG_NAMES):
        ax = axes[j]
        original_curve = logs_original[:, j].copy()
        original_curve[m[:, j] < 0.5] = np.nan
        ax.plot(original_curve, depth, label="observed")
        ax.plot(logs_completed[:, j], depth, linestyle="--", label="observed + imputed")

        miss_idx = np.where(m[:, j] < 0.5)[0]
        if len(miss_idx) > 0:
            # Shade contiguous missing intervals.
            splits = np.split(miss_idx, np.where(np.diff(miss_idx) != 1)[0] + 1)
            for seg in splits:
                if len(seg) > 0:
                    ax.axhspan(depth[seg.min()], depth[seg.max()], alpha=0.12)
        ax.set_title(name)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Depth")
    axes[-1].legend()
    fig.suptitle(f"Real-missing imputation case: {csv_path.name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[CASE] saved real-missing figure: {save_path}")


def run_train(train_files: List[Path], val_files: List[Path], test_files: List[Path]) -> None:
    print("Fitting scalers from train wells only...")
    log_scaler, depth_mean, depth_std, _ = collect_scaler_stats(train_files)

    print("Building train windows...")
    train_x, train_m, train_d, train_info, train_valid = build_windows_from_files(
        train_files, log_scaler, depth_mean, depth_std, split_name="train"
    )
    print("Building validation windows...")
    val_x, val_m, val_d, val_info, val_valid = build_windows_from_files(
        val_files, log_scaler, depth_mean, depth_std, split_name="val"
    )

    train_ds, train_loader = make_dataset_and_loader(train_x, train_m, train_d, BATCH_SIZE, shuffle=True)
    val_ds, val_loader = make_dataset_and_loader(val_x, val_m, val_d, BATCH_SIZE, shuffle=False)
    print(f"Window dataset: train={len(train_ds)}, val={len(val_ds)}")

    cddpm = build_cddpm_from_checkpoint_config()
    print("Training U-Net CDDPM...")
    train(cddpm, train_loader, val_loader, EPOCHS, LR)

    if Path(MODEL_PATH).exists():
        print(f"Loading best validation model: {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        cddpm.model.load_state_dict(state["model"])

    val_metrics = evaluate_imputation(cddpm, val_loader, max_batches=EVAL_BATCHES)
    print("Validation artificial-missing imputation metrics on standardized data:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.6f}")

    plot_test_example(cddpm, val_loader, str(Path(OUTPUT_DIR) / "unet_val_imputation_example.png"))

    torch.save({
        "model": cddpm.model.state_dict(),
        "log_scaler": log_scaler.state_dict(),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "log_names": LOG_NAMES,
        "window_size": WINDOW_SIZE,
        "diffusion_steps": DIFFUSION_STEPS,
        "split_file": SPLIT_FILE_PATH,
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "test_files": [p.name for p in test_files],
        "config": {
            "base_channels": BASE_CHANNELS,
            "time_dim": TIME_DIM,
            "depth_channels": DEPTH_CHANNELS,
        }
    }, FULL_CHECKPOINT_PATH)
    print(f"Full checkpoint saved to: {FULL_CHECKPOINT_PATH}")
    print("Train mode done. Test wells were not windowed or used during training.")


def run_test(test_files: List[Path]) -> None:
    cddpm, log_scaler, depth_mean, depth_std, checkpoint = load_trained_cddpm(LOAD_CHECKPOINT_PATH)

    print("Building test windows only...")
    test_x, test_m, test_d, test_info, test_valid = build_windows_from_files(
        test_files, log_scaler, depth_mean, depth_std, split_name="test"
    )
    test_ds, test_loader = make_dataset_and_loader(test_x, test_m, test_d, BATCH_SIZE, shuffle=False)
    print(f"Window dataset: test={len(test_ds)}")

    test_metrics = evaluate_imputation(cddpm, test_loader, max_batches=EVAL_BATCHES)
    print("Test artificial-missing imputation metrics on standardized data:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    plot_test_example(cddpm, test_loader, str(Path(OUTPUT_DIR) / "unet_test_imputation_example_loaded_model.png"))
    print("Test mode done. Real missing export is handled by --mode generate.")


def run_generate(test_files: List[Path], all_files: List[Path], case_wells: Optional[str], num_cases: int) -> None:
    cddpm, log_scaler, depth_mean, depth_std, checkpoint = load_trained_cddpm(LOAD_CHECKPOINT_PATH)
    selected = select_case_files(test_files, all_files, case_wells, num_cases)
    if len(selected) == 0:
        raise RuntimeError("No case wells selected for generation.")

    case_dir = Path(OUTPUT_DIR) / "case_figures"
    case_dir.mkdir(parents=True, exist_ok=True)
    print("Selected case wells:", [p.name for p in selected])

    for csv_path in selected:
        try:
            # 1) Export full-well real missing imputed CSV.
            impute_one_well_export(cddpm, csv_path, log_scaler, depth_mean, depth_std, OUTPUT_DIR)

            # 2) Plot artificial-missing comparison. This has ground truth.
            plot_artificial_mask_case_for_well(
                cddpm,
                csv_path,
                log_scaler,
                depth_mean,
                depth_std,
                str(case_dir / f"{csv_path.stem}_artificial_mask_compare.png"),
            )

            # 3) Plot real-missing imputation. This is qualitative only.
            plot_real_missing_case_for_well(
                cddpm,
                csv_path,
                log_scaler,
                depth_mean,
                depth_std,
                str(case_dir / f"{csv_path.stem}_real_missing_imputed.png"),
            )
        except Exception as e:
            print(f"[GENERATE-FAILED] {csv_path.name}: {e}")

    print("Generate mode done.")


def parse_args():
    parser = argparse.ArgumentParser(description="CDDPM 1D U-Net well-log imputation")
    parser.add_argument("--mode", choices=["train", "test", "generate", "all"], default=MODE,
                        help="train: train model; test: evaluate held-out test wells; generate: export/plot 1~3 case wells; all: train+test+generate")
    parser.add_argument("--csv-dir", default=CSV_DIR, help="folder containing per-well CSV files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="output folder")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path for test/generate; default is output_dir/cddpm_unet_full_checkpoint.pt")
    parser.add_argument("--case-wells", default=None,
                        help="comma-separated case well csv names or stems, e.g. well1.csv,well2")
    parser.add_argument("--num-cases", type=int, default=3, help="number of case wells to generate, recommended 1~3")
    parser.add_argument("--recreate-split", action="store_true", help="ignore cached split file and create a new 8:1:1 split")
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES, help="number of batches for artificial-missing evaluation")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    return parser.parse_args()


def apply_runtime_args(args) -> None:
    global CSV_DIR, OUTPUT_DIR, MODEL_PATH, FULL_CHECKPOINT_PATH, LOAD_CHECKPOINT_PATH, SPLIT_FILE_PATH
    global EVAL_BATCHES, EPOCHS, BATCH_SIZE

    CSV_DIR = args.csv_dir
    OUTPUT_DIR = args.output_dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    MODEL_PATH = str(Path(OUTPUT_DIR) / "cddpm_unet_model.pt")
    FULL_CHECKPOINT_PATH = str(Path(OUTPUT_DIR) / "cddpm_unet_full_checkpoint.pt")
    LOAD_CHECKPOINT_PATH = args.checkpoint if args.checkpoint else FULL_CHECKPOINT_PATH
    SPLIT_FILE_PATH = str(Path(OUTPUT_DIR) / "well_file_split.npz")

    EVAL_BATCHES = args.eval_batches
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size


def main():
    args = parse_args()
    apply_runtime_args(args)
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Mode:", args.mode)
    print("CSV_DIR:", CSV_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("CHECKPOINT:", LOAD_CHECKPOINT_PATH)

    train_files, val_files, test_files = split_csv_files_by_well(
        CSV_DIR,
        use_cache=not args.recreate_split,
        split_file=SPLIT_FILE_PATH,
    )
    all_files = scan_csv_files(CSV_DIR)

    if args.mode == "train":
        run_train(train_files, val_files, test_files)
    elif args.mode == "test":
        run_test(test_files)
    elif args.mode == "generate":
        run_generate(test_files, all_files, args.case_wells, min(max(args.num_cases, 1), 3))
    elif args.mode == "all":
        run_train(train_files, val_files, test_files)
        run_test(test_files)
        run_generate(test_files, all_files, args.case_wells, min(max(args.num_cases, 1), 3))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
