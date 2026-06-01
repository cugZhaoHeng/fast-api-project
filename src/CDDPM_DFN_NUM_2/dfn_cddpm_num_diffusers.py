import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGE_DIR = CURRENT_DIR / "images"
MODEL_DIR = CURRENT_DIR / "models"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.date_util import get_current_time
from utils.gpu_info import init_gpu_environment
from utils.logger import create_logger

logger = create_logger(__name__)

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 300
LR = 1e-4
TIMESTEPS = 500
GEN_GRID_SIZE = 3

DEVICE = init_gpu_environment()
DFN_DIR = DATA_DIR / "dfn_data" / "images"
CSV_PATH = DATA_DIR / "dfn_data" / "labels.csv"
LATEST_MODEL_PATH = MODEL_DIR / "latest_ddpm_dfn_num_mu_kappa_condition_model.pth"

DEFAULT_MIN_FRACTURES = 1
DEFAULT_MAX_FRACTURES = 19
DEFAULT_MIN_MEAN_MU = -90.0
DEFAULT_MAX_MEAN_MU = 90.0
DEFAULT_MIN_KAPPA = 1.0
DEFAULT_MAX_KAPPA = 10.0


def get_fracture_condition_range(csv_path: Path) -> Tuple[int, int]:
    if not csv_path.exists():
        return DEFAULT_MIN_FRACTURES, DEFAULT_MAX_FRACTURES

    df = pd.read_csv(csv_path, usecols=["num_fractures"])
    min_fractures = int(df["num_fractures"].min())
    max_fractures = int(df["num_fractures"].max())
    logger.info(f"min_fractures: {min_fractures}, max_fractures: {max_fractures}")
    return min_fractures, max_fractures


def get_mean_mu_range(csv_path: Path) -> Tuple[float, float]:
    if not csv_path.exists():
        return DEFAULT_MIN_MEAN_MU, DEFAULT_MAX_MEAN_MU

    df = pd.read_csv(csv_path, usecols=["mean_mu"])
    min_mu = float(df["mean_mu"].min())
    max_mu = float(df["mean_mu"].max())
    logger.info(f"observed mean_mu range: [{min_mu:.4f}, {max_mu:.4f}]")
    return min_mu, max_mu


def get_kappa_range(csv_path: Path) -> Tuple[float, float]:
    if not csv_path.exists():
        return DEFAULT_MIN_KAPPA, DEFAULT_MAX_KAPPA

    df = pd.read_csv(csv_path, usecols=["concentration_kappa"])
    min_kappa = float(df["concentration_kappa"].min())
    max_kappa = float(df["concentration_kappa"].max())
    logger.info(f"observed kappa range: [{min_kappa:.4f}, {max_kappa:.4f}]")
    return min_kappa, max_kappa


def format_condition_value(value: float) -> str:
    return f"{value:.2f}".replace("-", "neg").replace(".", "_")


def save_raw_image_grid(
    images: torch.Tensor,
    save_path: Path,
    grid_size: int,
    tile_size: int = IMAGE_SIZE,
) -> None:
    tiles = (images[:, 0].numpy() * 255.0).round().astype(np.uint8)
    grid = np.full(
        (grid_size * tile_size, grid_size * tile_size), 255, dtype=np.uint8
    )

    for idx, tile in enumerate(tiles):
        row = idx // grid_size
        col = idx % grid_size
        row_start = row * tile_size
        col_start = col * tile_size
        grid[row_start : row_start + tile_size, col_start : col_start + tile_size] = (
            tile
        )

    Image.fromarray(grid).save(save_path)


def save_preview_grid(
    images: torch.Tensor,
    save_path: Path,
    grid_size: int,
    fracture_count: int,
    mean_mu: float,
    kappa: float,
) -> None:
    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size * 2, grid_size * 2),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0)
    fig.suptitle(
        f"fracture number: {fracture_count}, mean_mu: {mean_mu:.2f}, kappa: {kappa:.2f}",
        color="white",
        fontsize=12,
        y=0.95,
    )
    fig.patch.set_facecolor("black")

    for idx, ax in enumerate(np.atleast_1d(axes).flat):
        ax.imshow(images[idx, 0], cmap="gray")
        ax.axis("off")

    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


class DFNDataset(Dataset):
    def __init__(self, image_dir: Path, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"])
        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        fracture_count = int(row["num_fractures"])
        mean_mu = float(row["mean_mu"])
        kappa = float(row["concentration_kappa"])
        return img, fracture_count, mean_mu, kappa


class ConditionalUNet(UNet2DModel):
    def __init__(
        self,
        min_fractures: int,
        max_fractures: int,
        min_mean_mu: float = DEFAULT_MIN_MEAN_MU,
        max_mean_mu: float = DEFAULT_MAX_MEAN_MU,
        min_kappa: float = DEFAULT_MIN_KAPPA,
        max_kappa: float = DEFAULT_MAX_KAPPA,
    ):
        num_class_embeds = max_fractures - min_fractures + 1

        super().__init__(
            sample_size=IMAGE_SIZE,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            num_class_embeds=num_class_embeds,
        )

        self.min_fractures = min_fractures
        self.max_fractures = max_fractures
        self.min_mean_mu = min_mean_mu
        self.max_mean_mu = max_mean_mu
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa

        self.time_embed_dim = self.time_embedding.linear_2.out_features
        self.kappa_log_min = math.log(max(min_kappa, 1e-6))
        self.kappa_log_max = math.log(max(max_kappa, min_kappa + 1e-6))

        # Mean fracture angle is continuous and periodic with 180-degree symmetry.
        self.mu_embedding = nn.Sequential(
            nn.Linear(2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Kappa controls concentration. Log-scaling makes this smoother to learn.
        self.kappa_embedding = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def _fracture_counts_to_class_labels(
        self,
        fracture_counts: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if fracture_counts is None:
            raise ValueError(
                "fracture_counts must be provided for conditional generation"
            )

        if not torch.is_tensor(fracture_counts):
            fracture_counts = torch.tensor(fracture_counts, device=device)

        fracture_counts = fracture_counts.to(device=device, dtype=torch.long)

        if fracture_counts.ndim == 0:
            fracture_counts = fracture_counts.unsqueeze(0)

        if fracture_counts.shape[0] == 1 and batch_size > 1:
            fracture_counts = fracture_counts.expand(batch_size)
        elif fracture_counts.shape[0] != batch_size:
            raise ValueError(
                f"Expected {batch_size} fracture counts, but received {fracture_counts.shape[0]}"
            )

        min_value = int(fracture_counts.min().item())
        max_value = int(fracture_counts.max().item())
        if min_value < self.min_fractures or max_value > self.max_fractures:
            raise ValueError(
                f"fracture_counts must be in [{self.min_fractures}, {self.max_fractures}], "
                f"but received values in [{min_value}, {max_value}]"
            )

        return fracture_counts - self.min_fractures

    def _prepare_continuous_condition(
        self,
        values: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        name: str,
        min_value: float,
        max_value: float,
    ) -> torch.Tensor:
        if values is None:
            raise ValueError(f"{name} must be provided for conditional generation")

        if not torch.is_tensor(values):
            values = torch.tensor(values, device=device)

        values = values.to(device=device, dtype=torch.float32)

        if values.ndim == 0:
            values = values.unsqueeze(0)

        if values.shape[0] == 1 and batch_size > 1:
            values = values.expand(batch_size)
        elif values.shape[0] != batch_size:
            raise ValueError(
                f"Expected {batch_size} {name} values, but received {values.shape[0]}"
            )

        actual_min = float(values.min().item())
        actual_max = float(values.max().item())
        tolerance = 1e-4
        if actual_min < min_value - tolerance or actual_max > max_value + tolerance:
            raise ValueError(
                f"{name} must be in [{min_value}, {max_value}], "
                f"but received values in [{actual_min}, {actual_max}]"
            )

        return values

    def _encode_mean_mus(self, mean_mus: torch.Tensor) -> torch.Tensor:
        mean_mus = mean_mus.to(dtype=self.dtype)
        mu_radians = mean_mus * torch.pi / 180.0
        periodic_radians = 2.0 * mu_radians
        mu_features = torch.stack(
            [torch.sin(periodic_radians), torch.cos(periodic_radians)],
            dim=-1,
        )
        return self.mu_embedding(mu_features)

    def _encode_kappas(self, kappas: torch.Tensor) -> torch.Tensor:
        kappas = kappas.to(dtype=self.dtype).clamp_min(1e-6)
        log_kappas = torch.log(kappas)
        denom = max(self.kappa_log_max - self.kappa_log_min, 1e-6)
        normalized = 2.0 * (log_kappas - self.kappa_log_min) / denom - 1.0
        return self.kappa_embedding(normalized.unsqueeze(-1))

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: Union[torch.Tensor, float, int],
        fracture_counts: Optional[torch.Tensor] = None,
        mean_mus: Optional[torch.Tensor] = None,
        kappas: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        class_labels = self._fracture_counts_to_class_labels(
            fracture_counts=fracture_counts,
            batch_size=noisy_images.shape[0],
            device=noisy_images.device,
        )
        mean_mus = self._prepare_continuous_condition(
            values=mean_mus,
            batch_size=noisy_images.shape[0],
            device=noisy_images.device,
            name="mean_mus",
            min_value=self.min_mean_mu,
            max_value=self.max_mean_mu,
        )
        kappas = self._prepare_continuous_condition(
            values=kappas,
            batch_size=noisy_images.shape[0],
            device=noisy_images.device,
            name="kappas",
            min_value=self.min_kappa,
            max_value=self.max_kappa,
        )

        sample = noisy_images
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )
        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        mu_emb = self._encode_mean_mus(mean_mus)
        kappa_emb = self._encode_kappas(kappas)
        emb = emb + class_emb + mu_emb + kappa_emb

        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)


def train_model(train_loader, model: ConditionalUNet, optimizer, noise_scheduler):
    loss_list = []
    start_epoch = 0
    end_epoch = start_epoch + NUM_EPOCHS
    train_times = 0
    total_training_time = 0.0
    start_time = time.time()

    if LATEST_MODEL_PATH.exists():
        checkpoint = torch.load(
            LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_list = checkpoint.get("train_losses", [])
        start_epoch = checkpoint["epoch"]
        end_epoch = start_epoch + NUM_EPOCHS
        train_times = checkpoint.get("train_times", 0)
        total_training_time = checkpoint.get("total_training_time", 0.0)
        logger.info(
            f"已加载 DDPM 模型, 起始 epoch={start_epoch}, 终止 epoch={end_epoch}"
        )
    else:
        logger.info("第一次训练 DDPM 模型")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        avg_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}", leave=False
        )

        for batch_idx, (images, fracture_count, mean_mu, kappa) in enumerate(progress_bar):
            images = images.to(DEVICE)
            fracture_count = fracture_count.to(DEVICE)
            mean_mu = mean_mu.to(DEVICE)
            kappa = kappa.to(DEVICE)

            noise = torch.randn_like(images)
            batch_size = images.shape[0]
            timesteps = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            noise_pred = model.forward(
                noisy_images,
                timesteps,
                fracture_counts=fracture_count,
                mean_mus=mean_mu,
                kappas=kappa,
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        loss_list.append(avg_loss)
        logger.info(f"Epoch [{epoch + 1}/{end_epoch}] 扩散模型 MSE Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    total_training_time += elapsed

    torch.save(
        {
            "epoch": end_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": loss_list,
            "train_times": train_times + 1,
            "total_training_time": total_training_time,
        },
        f=LATEST_MODEL_PATH,
    )
    logger.info(
        f"DDPM 模型已保存，本次耗时 {elapsed:.2f}s，累计总时长 {total_training_time:.2f}s"
    )


def generate(
    model: ConditionalUNet,
    target_n: str,
    target_mu: str,
    target_kappa: str,
):
    from src.CDDPM_DFN_NUM_2.evaluate_generated_dfn import evaluate_generated_grid

    fracture_count = int(target_n)
    mean_mu = float(target_mu)
    kappa = float(target_kappa)

    if fracture_count < model.min_fractures or fracture_count > model.max_fractures:
        logger.warning(
            f"裂缝数量必须在 {model.min_fractures}-{model.max_fractures} 之间"
        )
        return

    if mean_mu < model.min_mean_mu or mean_mu > model.max_mean_mu:
        logger.warning(
            f"mean_mu 必须在 {model.min_mean_mu}-{model.max_mean_mu} 之间"
        )
        return

    if kappa < model.min_kappa or kappa > model.max_kappa:
        logger.warning(f"kappa 必须在 {model.min_kappa}-{model.max_kappa} 之间")
        return

    if not LATEST_MODEL_PATH.exists():
        logger.warning("未找到模型文件，请先训练模型。")
        return

    num_images = GEN_GRID_SIZE**2
    fracture_counts = torch.full((num_images,), fracture_count, dtype=torch.long, device=DEVICE)
    mean_mus = torch.full((num_images,), mean_mu, dtype=torch.float32, device=DEVICE)
    kappas = torch.full((num_images,), kappa, dtype=torch.float32, device=DEVICE)

    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
    scheduler.set_timesteps(TIMESTEPS)

    images = torch.randn((num_images, 1, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)
    logger.info("开始生成图片")
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(
                images,
                t,
                fracture_counts=fracture_counts,
                mean_mus=mean_mus,
                kappas=kappas,
            ).sample

        images = scheduler.step(noise_pred, t, images).prev_sample

    images = ((images + 1) / 2).clamp(0, 1).cpu()

    timestamp = get_current_time()
    mu_tag = format_condition_value(mean_mu)
    kappa_tag = format_condition_value(kappa)

    raw_grid_path = (
        IMAGE_DIR
        / f"generated_dfn_grid_n_{fracture_count}_mu_{mu_tag}_kappa_{kappa_tag}_{timestamp}.png"
    )
    preview_path = (
        IMAGE_DIR
        / f"generated_dfn_preview_n_{fracture_count}_mu_{mu_tag}_kappa_{kappa_tag}_{timestamp}.png"
    )

    save_raw_image_grid(images, raw_grid_path, grid_size=GEN_GRID_SIZE)
    save_preview_grid(
        images,
        preview_path,
        grid_size=GEN_GRID_SIZE,
        fracture_count=fracture_count,
        mean_mu=mean_mu,
        kappa=kappa,
    )

    report = evaluate_generated_grid(
        image_path=raw_grid_path,
        tile_size=IMAGE_SIZE,
        grid_size=GEN_GRID_SIZE,
        target_count=fracture_count,
        target_mu=mean_mu,
        target_kappa=kappa,
        save_report=True,
    )

    summary = report["summary"]
    logger.info(f"图片生成完毕，原始网格保存至: {raw_grid_path.name}")
    logger.info(f"预览图保存至: {preview_path.name}")
    logger.info(
        "自动评估结果 -> "
        f"估计平均数量: {summary['mean_estimated_count']:.2f}, "
        f"估计主方向: {summary['overall_dominant_mu_deg']:.2f}°, "
        f"评估报告: {Path(report['report_path']).name}"
    )


def show_model_status():
    if not LATEST_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现模型文件。")
        return

    try:
        checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE)
        epoch = checkpoint.get("epoch", 0)
        train_times = checkpoint.get("train_times", 0)
        loss_list = checkpoint.get("train_losses", [])
        latest_loss = loss_list[-1] if loss_list else "N/A"
        total_training_time = checkpoint.get("total_training_time", 0)

        logger.info(f"{'DDPM 扩散模型状态报告':^36}")
        logger.info(f" 已训练总轮数:      {epoch}")
        logger.info(f" 累计训练次数:      {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f" 最近 MSE Loss:     {latest_loss:.4f}")
        duration = total_training_time / 60
        logger.info(f" 训练总时长:        {duration:.1f} 分钟")
        logger.info("=" * 40 + "\n")
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")


def generate_loss_plot():
    if not LATEST_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现模型文件，请先训练模型。")
        return

    try:
        checkpoint = torch.load(
            LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False
        )
        loss_list = checkpoint.get("train_losses", [])
        current_epoch = checkpoint.get("epoch", len(loss_list))

        if not loss_list:
            logger.warning("模型文件中没有发现损失记录数据。")
            return

        total_epochs = len(loss_list)
        num_points = min(total_epochs, 20)
        indices = np.linspace(0, total_epochs - 1, num_points, dtype=int)
        sampled_epochs = indices + 1
        sampled_losses = [loss_list[i] for i in indices]

        plt.figure(figsize=(8, 5))
        plt.plot(
            np.arange(1, total_epochs + 1),
            loss_list,
            color="blue",
            alpha=0.2,
            label="Full History",
        )
        plt.plot(
            sampled_epochs,
            sampled_losses,
            color="blue",
            linestyle="-",
            linewidth=2,
            label="Trend",
        )
        plt.scatter(
            sampled_epochs,
            sampled_losses,
            color="red",
            s=40,
            zorder=5,
            label="Sampled Points",
        )

        plt.title(f"DDPM Training Loss (Total Epochs: {total_epochs})")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        save_path = MODEL_DIR / f"dfn_loss_epoch_{current_epoch}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info("=" * 40)
        logger.info("Loss 图表已成功生成")
        logger.info(f"保存路径: {save_path.name}")
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"生成 Loss 图表失败: {e}")


def main():
    min_fractures, max_fractures = get_fracture_condition_range(CSV_PATH)
    observed_min_mu, observed_max_mu = get_mean_mu_range(CSV_PATH)
    observed_min_kappa, observed_max_kappa = get_kappa_range(CSV_PATH)

    model = ConditionalUNet(
        min_fractures=min_fractures,
        max_fractures=max_fractures,
        min_mean_mu=DEFAULT_MIN_MEAN_MU,
        max_mean_mu=DEFAULT_MAX_MEAN_MU,
        min_kappa=DEFAULT_MIN_KAPPA,
        max_kappa=DEFAULT_MAX_KAPPA,
    ).to(DEVICE)

    logger.info(
        f"mean_mu observed in dataset: [{observed_min_mu:.4f}, {observed_max_mu:.4f}], "
        f"supported input range: [{model.min_mean_mu:.1f}, {model.max_mean_mu:.1f}]"
    )
    logger.info(
        f"kappa observed in dataset: [{observed_min_kappa:.4f}, {observed_max_kappa:.4f}], "
        f"supported input range: [{model.min_kappa:.1f}, {model.max_kappa:.1f}]"
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      DDPM 扩散模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Generate)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 生成损失图表 (Loss Plot)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == "1":
            logger.info("开始加载 DFN 图片")
            dataset = DFNDataset(DFN_DIR, CSV_PATH)
            logger.info(f"DFN 图片加载完毕, 数量: {len(dataset)}")

            train_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
            )
            train_model(train_loader, model, optimizer, noise_scheduler)
        elif choice == "2":
            while True:
                logger.info("  [0/exit] 返回主菜单")
                target_n = input(
                    f"输入{model.min_fractures}-{model.max_fractures}之间的裂缝数量: "
                ).strip().lower()
                if target_n in ["0", "exit"]:
                    break

                target_mu = input(
                    f"输入{model.min_mean_mu:.1f}到{model.max_mean_mu:.1f}之间的 mean_mu: "
                ).strip().lower()
                if target_mu in ["0", "exit"]:
                    break

                target_kappa = input(
                    f"输入{model.min_kappa:.1f}到{model.max_kappa:.1f}之间的 kappa: "
                ).strip().lower()
                if target_kappa in ["0", "exit"]:
                    break

                try:
                    generate(model, target_n, target_mu, target_kappa)
                except ValueError:
                    logger.warning("请输入有效的裂缝数量、mean_mu 和 kappa 数值")
        elif choice == "3":
            show_model_status()
        elif choice == "4":
            generate_loss_plot()
        elif choice in ["0", "exit"]:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3, 4 或 0")


if __name__ == "__main__":
    main()
