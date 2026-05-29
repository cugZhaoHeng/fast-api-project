import os
from pathlib import Path
import sys
import time
from PIL import Image

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from diffusers import UNet2DModel, DDPMScheduler

from tqdm import tqdm
import matplotlib.pyplot as plt

# =====================================
# 参数
# =====================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGE_DIR = CURRENT_DIR / "images"
MODEL_DIR = CURRENT_DIR / "models"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "latest_ddpm_dfn_model.pth"

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger

logger = create_logger(__name__)
from utils.gpu_info import init_gpu_environment
from utils.date_util import get_current_time

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 500
LR = 1e-4
TIMESTEPS = 1000
DEVICE = init_gpu_environment()
DFN_DIR = DATA_DIR / "dfn_data" / "images"


# =====================================
# Dataset
# =====================================


class DFNDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(".png")
            ]
        )
        # 只取前10%用于快速训练
        num_images = len(self.image_paths)
        # self.image_paths = self.image_paths[: max(1, int(num_images * 0.1))]
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # 灰度图，1通道
        img = self.transform(img)
        return img


# =====================================
# Training
# =====================================


def train_model(train_loader, model, optimizer, noise_scheduler):
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
            f"已加载 DDPM 模型, 起始 epoch={start_epoch}，终止 epoch={end_epoch}"
        )
    else:
        logger.info("第一次训练 DDPM 模型")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        avg_loss = 0.0
        for batch_idx, images in enumerate(train_loader):
            images = images.to(DEVICE)
            noise = torch.randn_like(images)
            batch_size = images.shape[0]
            timesteps = torch.randint(0, TIMESTEPS, (batch_size,), device=DEVICE).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)

        loss_list.append(avg_loss)
        logger.info(
            f"Epoch [{epoch + 1}/{end_epoch}] 扩散模型 MSE Loss: {avg_loss:.4f}"
        )

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


def generate(model):
    # 加载本地保存的模型文件
    checkpoint = torch.load(LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    scheduler.set_timesteps(1000)

    images = torch.randn((16, 1, 128, 128), device=DEVICE)
    logger.info("开始生成图片")
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(images, t).sample

        images = scheduler.step(noise_pred, t, images).prev_sample
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    images = images.cpu()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"generated_dfn_{get_current_time()}.png")
    logger.info("图片生成完毕")


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
        logger.info(f" 已训练总轮数:    {epoch}")
        logger.info(f" 累计训练次数:    {train_times}")
        if isinstance(latest_loss, float):
            logger.info(f" 最近 MSE Loss:   {latest_loss:.4f} (越小说明去噪越准确)")
        duration = total_training_time / 60
        logger.info(f" 训练总时长：     {duration:.1f} 分钟")
        logger.info("=" * 40 + "\n")
    except Exception as e:
        logger.error(f"读取模型状态失败: {e}")


def generate_loss_plot():
    """模式: 从本地加载模型，绘制并保存损失函数图表 (通用版)"""
    if not LATEST_MODEL_PATH.exists():
        logger.info("\n[提示] 尚未发现模型文件，请先训练模型。")
        return

    try:
        # 加载权重字典
        checkpoint = torch.load(
            LATEST_MODEL_PATH, map_location=DEVICE, weights_only=False
        )
        loss_list = checkpoint.get("train_losses", [])
        current_epoch = checkpoint.get("epoch", len(loss_list))

        if not loss_list:
            logger.warning("模型文件中没有发现损失记录数据。")
            return

        total_epochs = len(loss_list)
        # 核心逻辑：最多提取 20 个采样点，防止图表点位过密
        num_points = min(total_epochs, 20)
        indices = np.linspace(0, total_epochs - 1, num_points, dtype=int)

        sampled_epochs = indices + 1  # 坐标从 1 开始
        sampled_losses = [loss_list[i] for i in indices]

        plt.figure(figsize=(8, 5))
        # 绘制浅色趋势线
        plt.plot(
            np.arange(1, total_epochs + 1),
            loss_list,
            color="blue",
            alpha=0.2,
            label="Full History",
        )
        # 绘制带点的采样线
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

        # 自动识别是 MNIST 还是 GEO，生成文件名
        prefix = "dfn"
        save_path = MODEL_DIR / f"{prefix}_loss_epoch_{current_epoch}.png"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info("=" * 40)
        logger.info(f"Loss 图表已成功生成！")
        logger.info(f"保存路径: {save_path.name}")
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"生成 Loss 图表失败: {e}")


def main():
    model = UNet2DModel(
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
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    while True:
        logger.info("\n" + "=" * 40)
        logger.info("      DDPM 扩散模型管理系统")
        logger.info(" [1] 训练模型 (Train)")
        logger.info(" [2] 测试模式 (Test: Generate/Denoising)")
        logger.info(" [3] 查看模型状态 (Status)")
        logger.info(" [4] 生成损失函数图表 (Loss Plot)")
        logger.info(" [0/exit] 退出程序")
        logger.info("=" * 40)

        choice = input("请选择主菜单功能: ").strip().lower()

        if choice == "1":
            # 1. 加载数据集（只在主进程执行一次）
            logger.info("开始加载DFN图片")
            dataset = DFNDataset(DFN_DIR)
            logger.info(f"DFN图片加载完毕, 数量: {len(dataset)}")

            train_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            train_model(train_loader, model, optimizer, noise_scheduler)
        elif choice == "2":
            while True:
                logger.info("\n  >>> 测试子菜单:")
                logger.info("  [1] Generate (生成手写数字图片)")
                logger.info("  [2] Denoising Process (去噪过程展示)")
                logger.info("  [0/exit] 返回主菜单")
                sub_choice = input("  请选择测试功能: ").strip().lower()
                if sub_choice == "1":
                    generate(model)
                elif sub_choice == "2":
                    logger.info("去噪展示待实现")
                elif sub_choice in ["0", "exit"]:
                    break
                else:
                    logger.info("  无效输入。")
        elif choice == "3":
            show_model_status()
        elif choice == "4":
            generate_loss_plot()
        elif choice in ["0", "exit"]:
            logger.info("退出程序...")
            break
        else:
            logger.info("无效输入，请重新输入 1, 2, 3 或 0")


if __name__ == "__main__":
    main()
