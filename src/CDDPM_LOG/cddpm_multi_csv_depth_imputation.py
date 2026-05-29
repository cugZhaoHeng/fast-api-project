# -*- coding: utf-8 -*-
"""
CDDPM well-log imputation with multi-CSV input and depth conditioning.

功能：
1. 读取一个目录下的多个 CSV 文件，每个 CSV 代表一口井。
2. 每个 CSV 至少包含：Depth, GR, RHOB, RILD, CNPOR。
3. 保留真实缺失值，构造 observed mask。
4. 每口井内部按 Depth 排序，并滑动窗口切分。
5. 深度条件使用两个通道：
   - relative depth: 窗口内 0~1
   - absolute depth: 全部井全局标准化后的真实深度
6. 80% 窗口训练，20% 窗口测试。
7. 训练时：从真实有值区域人工挖缺失段，loss 只计算人工 target 区域。
8. 测试时：人工遮挡测试，输出可视化结果。
9. 推理时：对真实缺失区域补全，并按每口井输出补全 CSV。

依赖：
pip install numpy pandas matplotlib torch scikit-learn

运行：
python cddpm_multi_csv_depth_imputation.py

请先修改下面 Config 里的 CSV_DIR / OUTPUT_DIR。
"""

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# 0. Config
# =========================
@dataclass
class Config:
    # 改成你的 CSV 文件目录
    CSV_DIR: str = "/home/tet/zhaoheng/fast-api-project/data/2023_log_csv"

    # 输出目录
    OUTPUT_DIR: str = "/home/tet/zhaoheng/fast-api-project/src/CDDPM_LOG/models"

    # 固定列名
    DEPTH_COL: str = "Depth"
    LOG_NAMES: Tuple[str, ...] = ("GR", "RHOB", "RILD", "CNPOR")

    # 滑动窗口参数
    WINDOW_SIZE: int = 128
    WINDOW_STRIDE: int = 32
    MIN_OBS_RATIO: float = 0.50       # 一个窗口总体观测比例太低则跳过
    MIN_TARGET_OBS_RATIO: float = 0.95 # 人工挖缺失段内，真实有值比例要求

    # 训练参数
    SEED: int = 42
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LR: float = 1e-3
    TRAIN_RATIO: float = 0.80

    # diffusion 参数
    DIFFUSION_STEPS: int = 200
    BETA_START: float = 1e-4
    BETA_END: float = 0.02

    # 人工 target 缺失段长度，单位是采样点
    TARGET_MIN_LEN: int = 16
    TARGET_MAX_LEN: int = 48
    MIN_TARGET_CHANNELS: int = 1
    MAX_TARGET_CHANNELS: int = 2

    # 模型参数
    HIDDEN_CHANNELS: int = 64
    TIME_DIM: int = 128
    DEPTH_CHANNELS: int = 16
    NUM_RES_BLOCKS: int = 4

    # 推理输出
    NUM_TEST_PLOTS: int = 8
    SAVE_REAL_IMPUTED_CSV: bool = True


CFG = Config()


# =========================
# 1. Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. Scaler
# =========================
class NanStandardScaler:
    """按列标准化，忽略 NaN。"""
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "NanStandardScaler":
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return x * self.std + self.mean


class DepthScaler:
    """深度全局标准化。"""
    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, depths: np.ndarray) -> "DepthScaler":
        self.mean = float(np.nanmean(depths))
        self.std = float(np.nanstd(depths))
        if self.std < 1e-6:
            self.std = 1.0
        return self

    def transform(self, depth: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("DepthScaler has not been fitted.")
        return (depth - self.mean) / self.std


# =========================
# 3. CSV loading and preprocessing
# =========================
def normalize_column_name(name: str) -> str:
    return str(name).strip().upper()


def find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    col_map = {normalize_column_name(c): c for c in df.columns}
    return col_map.get(normalize_column_name(target))


def read_one_csv(csv_path: Path, cfg: Config) -> Optional[pd.DataFrame]:
    """读取单井 CSV，只保留 Depth, GR, RHOB, RILD, CNPOR。缺列则返回 None。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[SKIP] {csv_path.name}: read failed: {e}")
        return None

    required = [cfg.DEPTH_COL] + list(cfg.LOG_NAMES)
    out = pd.DataFrame()

    for col in required:
        matched = find_column(df, col)
        if matched is None:
            print(f"[SKIP] {csv_path.name}: missing column {col}")
            return None
        out[col] = pd.to_numeric(df[matched], errors="coerce")

    out = out.dropna(subset=[cfg.DEPTH_COL])
    out = out.sort_values(cfg.DEPTH_COL).reset_index(drop=True)

    # 去掉重复深度，保留第一个
    out = out.drop_duplicates(subset=[cfg.DEPTH_COL], keep="first").reset_index(drop=True)

    if len(out) < cfg.WINDOW_SIZE:
        print(f"[SKIP] {csv_path.name}: too short, rows={len(out)}")
        return None

    return out


def basic_log_clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """基础异常值处理。注意：异常值置为 NaN，作为真实缺失。"""
    df = df.copy()

    # GR 通常不应为负，过大值通常为异常，可按需要调整
    df["GR"] = df["GR"].mask((df["GR"] < 0) | (df["GR"] > 400))

    # RHOB 常见范围约 1~4 g/cm3
    df["RHOB"] = df["RHOB"].mask((df["RHOB"] < 1.0) | (df["RHOB"] > 4.0))

    # RILD 电阻率必须大于 0，后面做 log10
    df["RILD"] = df["RILD"].mask(df["RILD"] <= 0)

    # CNPOR 有的数据是百分数，有的数据是小数；这里统一为小数 0~1
    # 如果当前井 CNPOR 中位数 > 1.5，认为是百分数
    if df["CNPOR"].dropna().shape[0] > 0 and df["CNPOR"].dropna().median() > 1.5:
        df["CNPOR"] = df["CNPOR"] / 100.0
    df["CNPOR"] = df["CNPOR"].mask((df["CNPOR"] < -0.2) | (df["CNPOR"] > 1.0))

    return df


def load_all_wells(cfg: Config) -> Dict[str, pd.DataFrame]:
    csv_dir = Path(cfg.CSV_DIR)
    csv_files = sorted(list(csv_dir.glob("*.csv")))
    print(f"发现 CSV 文件数量: {len(csv_files)}")

    wells: Dict[str, pd.DataFrame] = {}
    for path in csv_files:
        df = read_one_csv(path, cfg)
        if df is None:
            continue
        df = basic_log_clean(df, cfg)
        wells[path.stem] = df

    print(f"可用于后续处理的井数量: {len(wells)}")
    return wells


def fit_global_log_scaler(wells: Dict[str, pd.DataFrame], cfg: Config) -> NanStandardScaler:
    all_values = []
    for df in wells.values():
        values = df[list(cfg.LOG_NAMES)].values.astype(np.float32)
        # RILD 做 log10。因为 basic_log_clean 已经把 <=0 置 NaN。
        rild_idx = list(cfg.LOG_NAMES).index("RILD")
        values[:, rild_idx] = np.log10(values[:, rild_idx])
        all_values.append(values)

    all_values = np.concatenate(all_values, axis=0)
    scaler = NanStandardScaler().fit(all_values)
    print("log scaler mean:", scaler.mean)
    print("log scaler std :", scaler.std)
    return scaler


def fit_global_depth_scaler(wells: Dict[str, pd.DataFrame], cfg: Config) -> DepthScaler:
    all_depths = []
    for df in wells.values():
        all_depths.append(df[cfg.DEPTH_COL].values.astype(np.float32))
    all_depths = np.concatenate(all_depths, axis=0)
    scaler = DepthScaler().fit(all_depths)
    print(f"depth mean={scaler.mean:.4f}, std={scaler.std:.4f}")
    return scaler


def preprocess_one_well(
    df: pd.DataFrame,
    log_scaler: NanStandardScaler,
    depth_scaler: DepthScaler,
    cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    x_filled: [N, C]，标准化后，缺失填 0
    obs_mask: [N, C]，1=真实有值，0=真实缺失
    depth_raw: [N]
    depth_scaled: [N]
    """
    values = df[list(cfg.LOG_NAMES)].values.astype(np.float32)
    rild_idx = list(cfg.LOG_NAMES).index("RILD")
    values[:, rild_idx] = np.log10(values[:, rild_idx])

    obs_mask = np.isfinite(values).astype(np.float32)
    x_scaled = log_scaler.transform(values)
    x_filled = np.where(np.isfinite(x_scaled), x_scaled, 0.0).astype(np.float32)

    depth_raw = df[cfg.DEPTH_COL].values.astype(np.float32)
    depth_scaled = depth_scaler.transform(depth_raw).astype(np.float32)

    return x_filled, obs_mask, depth_raw, depth_scaled


# =========================
# 4. Window generation
# =========================
def make_windows_one_well(
    well_name: str,
    x: np.ndarray,
    mask: np.ndarray,
    depth_raw: np.ndarray,
    depth_scaled: np.ndarray,
    cfg: Config
):
    """
    输入：
    x: [N, C]
    mask: [N, C]
    depth_raw: [N]
    depth_scaled: [N]

    输出：
    x_windows: [W, C, L]
    m_windows: [W, C, L]
    d_windows: [W, 2, L]，第0通道相对深度，第1通道绝对深度标准化
    metadata: list，每个窗口记录井名、start、end、raw depth
    """
    x_windows = []
    m_windows = []
    d_windows = []
    metadata = []

    n = len(x)
    L = cfg.WINDOW_SIZE

    for start in range(0, n - L + 1, cfg.WINDOW_STRIDE):
        end = start + L
        xw = x[start:end]          # [L, C]
        mw = mask[start:end]       # [L, C]
        dr = depth_raw[start:end]  # [L]
        ds = depth_scaled[start:end] # [L]

        if mw.mean() < cfg.MIN_OBS_RATIO:
            continue

        rel_depth = np.linspace(0.0, 1.0, L, dtype=np.float32)
        depth_feat = np.stack([rel_depth, ds.astype(np.float32)], axis=0) # [2, L]

        x_windows.append(xw)
        m_windows.append(mw)
        d_windows.append(depth_feat)
        metadata.append({
            "well": well_name,
            "start_idx": start,
            "end_idx": end,
            "depth_start": float(dr[0]),
            "depth_end": float(dr[-1]),
        })

    if len(x_windows) == 0:
        return None, None, None, []

    x_windows = np.stack(x_windows, axis=0)  # [W, L, C]
    m_windows = np.stack(m_windows, axis=0)  # [W, L, C]
    d_windows = np.stack(d_windows, axis=0)  # [W, 2, L]

    x_windows = np.transpose(x_windows, (0, 2, 1)).astype(np.float32) # [W, C, L]
    m_windows = np.transpose(m_windows, (0, 2, 1)).astype(np.float32) # [W, C, L]
    d_windows = d_windows.astype(np.float32)

    return x_windows, m_windows, d_windows, metadata


def build_all_windows(wells: Dict[str, pd.DataFrame], cfg: Config):
    log_scaler = fit_global_log_scaler(wells, cfg)
    depth_scaler = fit_global_depth_scaler(wells, cfg)

    all_x, all_m, all_d, all_meta = [], [], [], []

    for well_name, df in wells.items():
        x, m, d_raw, d_scaled = preprocess_one_well(df, log_scaler, depth_scaler, cfg)
        xw, mw, dw, meta = make_windows_one_well(well_name, x, m, d_raw, d_scaled, cfg)
        if xw is None:
            print(f"[NO WINDOWS] {well_name}")
            continue
        all_x.append(xw)
        all_m.append(mw)
        all_d.append(dw)
        all_meta.extend(meta)
        print(f"[WINDOWS] {well_name}: {xw.shape[0]}")

    if len(all_x) == 0:
        raise RuntimeError("没有生成任何窗口，请检查数据、窗口长度、观测比例阈值。")

    all_x = np.concatenate(all_x, axis=0)
    all_m = np.concatenate(all_m, axis=0)
    all_d = np.concatenate(all_d, axis=0)

    print("all_x shape:", all_x.shape)
    print("all_m shape:", all_m.shape)
    print("all_d shape:", all_d.shape)
    print("window observed ratio:", float(all_m.mean()))

    return all_x, all_m, all_d, all_meta, log_scaler, depth_scaler


# =========================
# 5. Dataset
# =========================
class WellLogWindowDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, obs_masks: np.ndarray, depth_windows: np.ndarray, cfg: Config):
        self.x = torch.from_numpy(x_windows).float()       # [N, C, L]
        self.m_obs = torch.from_numpy(obs_masks).float()   # [N, C, L]
        self.depth = torch.from_numpy(depth_windows).float() # [N, 2, L]
        self.cfg = cfg

    def __len__(self):
        return len(self.x)

    def make_target_mask(self, x0: torch.Tensor, m_obs: torch.Tensor) -> torch.Tensor:
        C, L = x0.shape
        m_ta = torch.zeros_like(x0)

        num_ch = random.randint(self.cfg.MIN_TARGET_CHANNELS, self.cfg.MAX_TARGET_CHANNELS)
        num_ch = min(num_ch, C)
        channels = random.sample(range(C), k=num_ch)

        for ch in channels:
            seg_len = random.randint(self.cfg.TARGET_MIN_LEN, self.cfg.TARGET_MAX_LEN)
            seg_len = min(seg_len, L)

            found = False
            for _ in range(50):
                start = random.randint(0, L - seg_len)
                end = start + seg_len
                if m_obs[ch, start:end].mean().item() >= self.cfg.MIN_TARGET_OBS_RATIO:
                    m_ta[ch, start:end] = 1.0
                    found = True
                    break

            # 找不到连续高观测段，则退化为从该曲线真实有值位置中随机选点
            if not found:
                valid_idx = torch.where(m_obs[ch] > 0.5)[0]
                if len(valid_idx) > 0:
                    k = min(seg_len, len(valid_idx))
                    perm = torch.randperm(len(valid_idx))[:k]
                    chosen = valid_idx[perm]
                    m_ta[ch, chosen] = 1.0

        return m_ta

    def __getitem__(self, idx):
        x0 = self.x[idx].clone()
        m_obs = self.m_obs[idx].clone()
        depth = self.depth[idx].clone()

        m_ta = self.make_target_mask(x0, m_obs)

        # m_cond = 1 表示模型可见；真实缺失和人工 target 都不可见
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


# =========================
# 6. Model
# =========================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


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
    x_ta_t: [B, C, L]，目标区域带噪数据，非目标区域为 0
    x_cond: [B, C, L]，条件数据，真实缺失和人工 target 为 0
    m_cond: [B, C, L]，条件 mask
    depth:  [B, 2, L]，relative depth + standardized absolute depth
    t:      [B]
    输出：
    pred_noise: [B, C, L]
    """
    def __init__(
        self,
        log_channels: int,
        hidden_channels: int,
        time_dim: int,
        depth_channels: int,
        num_res_blocks: int,
    ):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.depth_proj = nn.Sequential(
            nn.Conv1d(2, depth_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(depth_channels, depth_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )

        in_channels = log_channels * 3 + depth_channels
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList([
            ConditionalResidualBlock1D(hidden_channels, time_dim)
            for _ in range(num_res_blocks)
        ])

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
        depth: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        d_emb = self.depth_proj(depth)

        x = torch.cat([x_ta_t, x_cond, m_cond, d_emb], dim=1)
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


# =========================
# 7. CDDPM
# =========================
class CDDPM:
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: str,
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
        depth = batch["depth"].to(self.device)

        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        xt, noise = self.q_sample(x0, t)

        # 只把人工 target 区域的 noisy 数据输入模型
        x_ta_t = xt * m_ta

        pred_noise = self.model(x_ta_t, x_cond, m_cond, depth, t)

        denom = m_ta.sum().clamp(min=1.0)
        loss = (((pred_noise - noise) ** 2) * m_ta).sum() / denom
        return loss

    @torch.no_grad()
    def impute(
        self,
        x0_filled: torch.Tensor,
        m_obs: torch.Tensor,
        depth: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        keep_observed: bool = True,
    ) -> torch.Tensor:
        """
        x0_filled: [B, C, L]
        m_obs: [B, C, L]，1=真实有值，0=真实缺失
        depth: [B, 2, L]
        target_mask:
            None: 补真实缺失区域 1 - m_obs
            非 None: 补指定区域，比如人工遮挡区域
        """
        self.model.eval()
        x0_filled = x0_filled.to(self.device)
        m_obs = m_obs.to(self.device)
        depth = depth.to(self.device)

        if target_mask is None:
            m_target = 1.0 - m_obs
        else:
            m_target = target_mask.to(self.device)

        # 条件区域：不是 target 的地方都作为条件。
        # 若 target 是真实缺失，m_cond = m_obs。
        m_cond = 1.0 - m_target
        if keep_observed:
            m_cond = m_cond * m_obs
        x_cond = x0_filled * m_cond

        x = torch.randn_like(x0_filled) * m_target

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, x_cond, m_cond, depth, t)

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

            x = x_prev * m_target

        x_completed = x_cond + x * m_target
        # 如果有一些既不是条件也不是 target 的真实缺失位置，则仍保持生成/0，这里一般不会出现。
        return x_completed


# =========================
# 8. Train / Evaluate
# =========================
def train_model(cddpm: CDDPM, train_loader: DataLoader, cfg: Config, output_dir: Path):
    optimizer = torch.optim.Adam(cddpm.model.parameters(), lr=cfg.LR)
    loss_history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        cddpm.model.train()
        total_loss = 0.0
        total_count = 0

        for batch in train_loader:
            loss = cddpm.p_losses(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cddpm.model.parameters(), 1.0)
            optimizer.step()

            bs = batch["x0"].size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch:03d}/{cfg.EPOCHS} | loss={avg_loss:.6f}")

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    return loss_history


def inverse_logs(x_scaled_cl: np.ndarray, log_scaler: NanStandardScaler, cfg: Config) -> np.ndarray:
    """
    x_scaled_cl: [C, L]
    返回原始量纲 [C, L]，其中 RILD 从 log10 还原。
    """
    x_lc = x_scaled_cl.T  # [L, C]
    inv = log_scaler.inverse_transform(x_lc).astype(np.float32)
    rild_idx = list(cfg.LOG_NAMES).index("RILD")
    inv[:, rild_idx] = np.power(10.0, inv[:, rild_idx])
    return inv.T


@torch.no_grad()
def evaluate_artificial_mask(
    cddpm: CDDPM,
    test_dataset: Dataset,
    log_scaler: NanStandardScaler,
    cfg: Config,
    output_dir: Path,
):
    """在测试集上重新人工遮挡，计算 target 区域 MAE/RMSE，并保存少量图。"""
    cddpm.model.eval()
    device = cddpm.device

    n = min(cfg.NUM_TEST_PLOTS, len(test_dataset))
    indices = np.random.choice(len(test_dataset), size=n, replace=False) if n > 0 else []

    maes = []
    rmses = []

    for plot_i, idx in enumerate(indices):
        sample = test_dataset[int(idx)]
        x0 = sample["x0"].unsqueeze(0).to(device)
        m_obs = sample["m_obs"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        m_ta = sample["m_ta"].unsqueeze(0).to(device)

        x_imputed = cddpm.impute(x0, m_obs, depth, target_mask=m_ta, keep_observed=True)

        target = m_ta.cpu().numpy()[0]
        orig = x0.cpu().numpy()[0]
        imp = x_imputed.cpu().numpy()[0]

        diff = (imp - orig) * target
        denom = target.sum()
        if denom > 0:
            mae = np.abs(diff).sum() / denom
            rmse = np.sqrt((diff ** 2).sum() / denom)
            maes.append(mae)
            rmses.append(rmse)

        # 画原始量纲
        orig_inv = inverse_logs(orig, log_scaler, cfg)
        imp_inv = inverse_logs(imp, log_scaler, cfg)
        cond = orig.copy()
        cond[target > 0.5] = np.nan
        cond_inv = inverse_logs(np.nan_to_num(cond, nan=0.0), log_scaler, cfg)
        cond_inv[target > 0.5] = np.nan

        depth_axis = np.arange(orig.shape[1])
        fig, axes = plt.subplots(1, len(cfg.LOG_NAMES), figsize=(15, 6), sharey=True)
        for c, name in enumerate(cfg.LOG_NAMES):
            ax = axes[c]
            ax.plot(orig_inv[c], depth_axis, label="Original")
            ax.plot(cond_inv[c], depth_axis, label="Condition")
            ax.plot(imp_inv[c], depth_axis, linestyle="--", label="Imputed")
            masked_idx = np.where(target[c] > 0.5)[0]
            if len(masked_idx) > 0:
                ax.axhspan(masked_idx.min(), masked_idx.max(), alpha=0.15)
            ax.set_title(name)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Depth index in window")
        axes[-1].legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"test_artificial_impute_{plot_i:03d}.png", dpi=160)
        plt.close()

    if len(maes) > 0:
        print(f"Test artificial mask MAE : {float(np.mean(maes)):.6f}")
        print(f"Test artificial mask RMSE: {float(np.mean(rmses)):.6f}")
    else:
        print("Test artificial mask: no valid target mask generated.")


# =========================
# 9. Real missing imputation for each well
# =========================
def make_sequential_windows_for_one_well(
    well_name: str,
    x: np.ndarray,
    mask: np.ndarray,
    depth_raw: np.ndarray,
    depth_scaled: np.ndarray,
    cfg: Config
):
    """用于整井补全，尽量覆盖全部深度。末尾不足窗口时补最后一个窗口。"""
    n = len(x)
    L = cfg.WINDOW_SIZE
    starts = list(range(0, max(n - L + 1, 1), cfg.WINDOW_STRIDE))
    if n >= L and starts[-1] != n - L:
        starts.append(n - L)

    xws, mws, dws, ranges = [], [], [], []
    for start in starts:
        end = start + L
        if end > n:
            continue
        xw = x[start:end]
        mw = mask[start:end]
        ds = depth_scaled[start:end]
        rel_depth = np.linspace(0.0, 1.0, L, dtype=np.float32)
        depth_feat = np.stack([rel_depth, ds.astype(np.float32)], axis=0)

        xws.append(xw.T.astype(np.float32))
        mws.append(mw.T.astype(np.float32))
        dws.append(depth_feat.astype(np.float32))
        ranges.append((start, end))

    if len(xws) == 0:
        return None, None, None, []

    return np.stack(xws), np.stack(mws), np.stack(dws), ranges


@torch.no_grad()
def impute_real_missing_all_wells(
    cddpm: CDDPM,
    wells: Dict[str, pd.DataFrame],
    log_scaler: NanStandardScaler,
    depth_scaler: DepthScaler,
    cfg: Config,
    output_dir: Path,
):
    real_out_dir = output_dir / "real_missing_imputed_csv"
    real_out_dir.mkdir(parents=True, exist_ok=True)

    device = cddpm.device
    cddpm.model.eval()

    for well_name, df in wells.items():
        x, m, d_raw, d_scaled = preprocess_one_well(df, log_scaler, depth_scaler, cfg)
        xw, mw, dw, ranges = make_sequential_windows_for_one_well(well_name, x, m, d_raw, d_scaled, cfg)
        if xw is None:
            continue

        # 累积多个窗口补全结果；重叠区域取平均
        pred_sum = np.zeros_like(x, dtype=np.float32)   # [N, C]
        pred_cnt = np.zeros_like(x, dtype=np.float32)   # [N, C]

        bs = cfg.BATCH_SIZE
        for i in range(0, len(xw), bs):
            xb = torch.from_numpy(xw[i:i+bs]).float().to(device)
            mb = torch.from_numpy(mw[i:i+bs]).float().to(device)
            db = torch.from_numpy(dw[i:i+bs]).float().to(device)

            # target_mask=None => 只补真实缺失区域
            completed = cddpm.impute(xb, mb, db, target_mask=None, keep_observed=True)
            completed_np = completed.cpu().numpy() # [B, C, L]

            for j in range(completed_np.shape[0]):
                start, end = ranges[i + j]
                pred_sum[start:end] += completed_np[j].T
                pred_cnt[start:end] += 1.0

        pred_cnt[pred_cnt < 1.0] = 1.0
        completed_scaled = pred_sum / pred_cnt

        # 观测位置强制使用原始 x，真实缺失位置使用模型补全
        completed_scaled = x * m + completed_scaled * (1.0 - m)

        # 还原原始量纲
        completed_original = log_scaler.inverse_transform(completed_scaled).astype(np.float32)
        rild_idx = list(cfg.LOG_NAMES).index("RILD")
        completed_original[:, rild_idx] = np.power(10.0, completed_original[:, rild_idx])

        out = pd.DataFrame()
        out[cfg.DEPTH_COL] = df[cfg.DEPTH_COL].values
        for ci, name in enumerate(cfg.LOG_NAMES):
            out[name] = completed_original[:, ci]
            out[name + "_was_missing"] = (m[:, ci] < 0.5).astype(int)

        out_path = real_out_dir / f"{well_name}_imputed.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[REAL IMPUTED] {well_name} -> {out_path.name}")


# =========================
# 10. Main
# =========================
def main():
    set_seed(CFG.SEED)
    output_dir = Path(CFG.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    wells = load_all_wells(CFG)
    if len(wells) == 0:
        raise RuntimeError("没有可用 CSV。请检查 CSV_DIR 和列名。")

    x_windows, m_windows, d_windows, meta, log_scaler, depth_scaler = build_all_windows(wells, CFG)

    dataset = WellLogWindowDataset(x_windows, m_windows, d_windows, CFG)
    train_size = int(len(dataset) * CFG.TRAIN_RATIO)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(CFG.SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    print(f"train windows: {len(train_dataset)}, test windows: {len(test_dataset)}")

    model = ConditionalDenoiser1D(
        log_channels=len(CFG.LOG_NAMES),
        hidden_channels=CFG.HIDDEN_CHANNELS,
        time_dim=CFG.TIME_DIM,
        depth_channels=CFG.DEPTH_CHANNELS,
        num_res_blocks=CFG.NUM_RES_BLOCKS,
    )

    cddpm = CDDPM(
        model=model,
        timesteps=CFG.DIFFUSION_STEPS,
        beta_start=CFG.BETA_START,
        beta_end=CFG.BETA_END,
        device=device,
    )

    print("Start training...")
    train_model(cddpm, train_loader, CFG, output_dir)

    ckpt_path = output_dir / "cddpm_multi_csv_depth_model.pt"
    torch.save({
        "model_state_dict": cddpm.model.state_dict(),
        "config": CFG.__dict__,
        "log_scaler_mean": log_scaler.mean,
        "log_scaler_std": log_scaler.std,
        "depth_scaler_mean": depth_scaler.mean,
        "depth_scaler_std": depth_scaler.std,
    }, ckpt_path)
    print("Saved model:", ckpt_path)

    print("Evaluate artificial missing on test windows...")
    evaluate_artificial_mask(cddpm, test_dataset, log_scaler, CFG, output_dir)

    if CFG.SAVE_REAL_IMPUTED_CSV:
        print("Impute real missing values for all wells...")
        impute_real_missing_all_wells(cddpm, wells, log_scaler, depth_scaler, CFG, output_dir)

    print("Done. Output dir:", output_dir)


if __name__ == "__main__":
    main()
