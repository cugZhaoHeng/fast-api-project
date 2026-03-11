import os
from huggingface_hub import snapshot_download

# 1. 设置镜像站（非常重要，确保下载速度且不报错）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 定义模型 ID 和 本地保存路径
repo_id = "xkronosx/ddpm-mnist-32"
local_dir = "D:/my_models/ddpm-mnist-32"

print(f"开始从镜像站下载模型 {repo_id} 到 {local_dir}...")

try:
    # 3. 调用下载函数
    # local_dir_use_symlinks=False 是关键，Windows 必须设为 False 才能得到真实文件
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=False  # 该模型是公开的，不需要 Token
    )
    print(f"\n下载成功！模型已完整保存在：{path}")
    print("文件夹结构检查：")
    for root, dirs, files in os.walk(local_dir):
        level = root.replace(local_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

except Exception as e:
    print(f"\n下载失败，错误原因: {e}")