from diffusers import AutoencoderKL

# 直接从 Hugging Face Hub 下载（自动处理路径）
vae = AutoencoderKL.from_pretrained(
    "madebyollin/taesd",
    subfolder="vae"  # 关键：指定子文件夹
)
vae.config["latent_channels"] = 4  # 适配 128x128
vae.to("cuda")