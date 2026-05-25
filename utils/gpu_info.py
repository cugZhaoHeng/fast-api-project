import os
from pathlib import Path
import sys
import torch
import pynvml
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger

logger = create_logger(__name__)

def init_gpu_environment():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    # 自动寻找显存剩余最多的卡
    best_gpu_index = 0
    max_free_mem = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.free > max_free_mem:
            max_free_mem = info.free
            best_gpu_index = i

    # 1. 设置环境变量（物理隔离）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)

    # 2. 只有设置完环境变量后，才能初始化 torch.device
    device = torch.device("cuda:0")

    # 3. 清理本进程之前的残留（如果不小心启动过）
    torch.cuda.empty_cache()

    # 4. 打印最终确认信息
    logger.info("==========================================")
    logger.info("设备初始化成功:")
    logger.info(f"  物理显卡 ID (nvidia-smi): {best_gpu_index}")
    logger.info(f"  逻辑显卡 ID (pytorch):    {device}")
    logger.info(f"  当前可用显存: {max_free_mem / 1024 ** 2:.0f} MB")
    logger.info(f"  显卡型号: {torch.cuda.get_device_name(0)}")
    logger.info("  提示: empty_cache() 已执行，仅清理本进程缓存。")
    logger.info("==========================================")

    pynvml.nvmlShutdown()
    return device