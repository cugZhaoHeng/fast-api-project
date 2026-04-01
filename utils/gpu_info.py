import os
import torch
import pynvml


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
    print(f"==========================================")
    print(f"设备初始化成功:")
    print(f"  物理显卡 ID (nvidia-smi): {best_gpu_index}")
    print(f"  逻辑显卡 ID (pytorch):    {device}")
    print(f"  当前可用显存: {max_free_mem / 1024 ** 2:.0f} MB")
    print(f"  显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"  提示: empty_cache() 已执行，仅清理本进程缓存。")
    print(f"==========================================")

    pynvml.nvmlShutdown()
    return device


# 执行初始化
device = init_gpu_environment()