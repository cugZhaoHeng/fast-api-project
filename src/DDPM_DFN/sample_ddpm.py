import torch
import matplotlib.pyplot as plt

from diffusers import (
    UNet2DModel,
    DDPMScheduler
)

IMAGE_SIZE = 128

DEVICE = "cuda"

model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64,128,256,512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "./models/ddpm/epoch_100.pth"
    )
)

model.eval()

scheduler = DDPMScheduler(
    num_train_timesteps=1000
)

scheduler.set_timesteps(1000)

images = torch.randn(
    (16,1,128,128),
    device=DEVICE
)

for t in scheduler.timesteps:

    with torch.no_grad():

        noise_pred = model(
            images,
            t
        ).sample

    images = scheduler.step(
        noise_pred,
        t,
        images
    ).prev_sample

images = (images + 1) / 2

images = images.clamp(0,1)

images = images.cpu()

fig, axes = plt.subplots(4,4,figsize=(8,8))

for i, ax in enumerate(axes.flat):

    ax.imshow(
        images[i,0],
        cmap="gray"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig("generated_dfn.png")

plt.show()