import os

import torch
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt

# 经过验证，这个 ID 目前是公开可用的
model_id = "xkronosx/ddpm-mnist-32"

def download_model():
    local_path = "./models/ddpm-mnist-local"  # 你可以自定义路径

    # 2. 如果本地还没下载过，则下载并保存
    if not os.path.exists(local_path):
        print(f"正在从网上下载模型并保存到 {local_path}...")

        pipe = DDPMPipeline.from_pretrained(model_id)
        pipe.save_pretrained(local_path)
        print("模型已成功保存到本地。")
    else:
        print(f"本地路径 {local_path} 已存在，无需重复下载。")

# 指向包含 model_index.json 的那个根目录
local_root_path = "D:/my_models/ddpm-mnist-32"

def run_standard_pipeline():
    try:
        print("正在从完整 Pipeline 结构加载模型...")
        # 只要有 model_index.json，这一行就能自动识别并加载所有组件
        pipe = DDPMPipeline.from_pretrained(
            local_root_path,
            local_files_only=True,
            use_safetensors=True
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        print("加载成功！开始生成图片...")
        # 注意：因为 scheduler 里的训练步数是 200，所以生成时通常也用 200 步
        result = pipe(batch_size=16, num_inference_steps=200)
        images = result.images

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        plt.show()

    except Exception as e:
        print(f"运行失败: {e}")

if __name__ == "__main__":
    run_standard_pipeline()