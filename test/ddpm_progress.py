# 当前代码是为了测试DDPM的加噪的公式，图片使用MNIST，然后实现逐步加噪，期间不涉及任何的神经网络，只是简单的给图片加上噪声
from math import sqrt
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
BATCH_SIZE=10
os.makedirs(IMAGE_DIR, exist_ok=True)

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__) 
from utils.date_util import get_current_time
# 1. 读取图片
transform = transforms.Compose([
    transforms.ToTensor(),# ToTensor()函数，将原本的PIL.Image.Image类转化为Tensor类，数值从0-255转化为0-1,
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # 将原本0-1的张量，转化为[-1, 1]，使用均值为0.5，标准差为0.5的线性变换， output = (x-mean) / std, 扩散模型在工业界默认使用的数值就是 [-1, 1]，因为扩散模型需要假设数据具有零均值、单位方差
])
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 这里要用元组来接收
images, labels = next(iter(train_loader))
# images 的形状是 [128, 1, 28, 28]
logger.info(f"images shape: {images[0].shape}")
# 2. 将图片变成张量
current_image = images[0].squeeze()
# 3. 给图片添加同等形状的噪声
timesteps = 9
betas = torch.linspace(0.01,0.2,timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
logger.info(f"betas:{betas}")
logger.info(f"alphas: {alphas}")
logger.info(f"alphas_cumprod: {alphas_cumprod}")
logger.info(f"alphas_cumprod_prev: {alphas_cumprod_prev}")

noise = torch.randn_like(current_image)
beta = betas[2]
x = beta*noise  + (1-beta) * current_image
# 4. 定义 beta 列表，1000步，从0-1

# current_beta = 0.1
# # 这里的axes是一个二维数组，3*3
# fig, axes = plt.subplots(1, timesteps,figsize=(10* timesteps,10))
# axes = axes.flatten()
# # 一个迭代的加噪过程，加噪的时候，既可以使用相同的beta值，如0.1， 也可以根据加噪的时间步，使用动态的beta，
# for i in range(0, timesteps):
#     noise_i = torch.randn_like(current_image)
#     current_image = sqrt(alphas[i]) * current_image + sqrt((1-alphas[i])) * noise_i
#     axes[i].axis('off')
#     axes[i].imshow((current_image + 1) /2, cmap='gray')
# plt.tight_layout()
# plt.savefig(IMAGE_DIR / str(get_current_time()))
# plt.close() 

# image_0 = images[0].squeeze()
# fig, axes = plt.subplots(1, timesteps,figsize=(10* timesteps,10))
# axes = axes.flatten()
# for i in range(0, timesteps):
#     alpha_i = alphas_cumprod_prev[i]
#     noise_i = torch.randn_like(image_0)
#     output_i = sqrt(alpha_i) * image_0 + sqrt(1-alpha_i)*noise_i
#     axes[i].axis('off')
#     axes[i].imshow((output_i + 1) / 2, cmap='gray')
# plt.tight_layout()
# plt.savefig(IMAGE_DIR / str(get_current_time()))
# 一步到位的加噪
# image_1 = images[1].squeeze()
# alpha_1 = alphas_cumprod[timesteps - 1]
# current_image_1 = sqrt(alpha_1) * image_1 + sqrt(1-alpha_1)*noise_i
# plt.imshow(current_image_1, cmap='gray')
# plt.savefig(IMAGE_DIR / str(get_current_time()))

# 绘制图片
# fig, axes = plt.subplots(1,2,figsize=(10,5)) 
# axes[0].imshow(current_image, cmap='gray')
# axes[1].imshow(x, cmap='gray')
# plt.show()
# plt.savefig(IMAGE_DIR / str(get_current_time()))

# 编写去噪的代码，只记录一步，比如，从原图加一步噪声，然后记录下噪声，然后根据去噪公式，恢复原图
image_0 = images[0].squeeze()
noise = torch.randn_like(image_0)
beta = betas[timesteps-1]
alpha = 1.0 - beta
out_put = sqrt(alpha) * image_0 + sqrt(1-alpha)*noise

out_put_d = (1/sqrt(alpha))*(out_put - (beta/sqrt(1-alpha) * noise))
logger.info(f"image_0:{image_0}")
logger.info(f"out_put_d: {out_put_d}")
# 怎么计算两个输出之间的差距呢？用什么函数？可以使用 MSE， RMSE
# L1损失，L1范数，这里需要搞清楚一个概念，范数是范数，是一个多维的张量，而损失是一个值，要么求和，要么计算平均值，只能是一个标量
L1_loss = torch.abs(image_0 - out_put_d)
logger.info(f"L1范数：{L1_loss}")
logger.info(f"L1损失：{torch.sum(L1_loss)}")
MAE = torch.mean(L1_loss)
logger.info(f"MAE:{MAE}")

# L2损失，L2范数
L2_loss = torch.norm(image_0 - out_put_d)
logger.info(f"L2范数：{L2_loss}")
logger.info(f"L2损失：{torch.sum(L2_loss)}")
MSE = torch.mean((image_0 - out_put_d) ** 2)
logger.info(f"MSE：{MSE}")
RMSE = torch.sqrt(MSE)
logger.info(f"RMSE:{RMSE}")


# fig, axes = plt.subplots(1,3,figsize=(15,5)) 
# axes[0].imshow(image_0, cmap='gray')
# axes[1].imshow(out_put, cmap='gray')
# axes[2].imshow(out_put_d, cmap='gray')
# plt.show()
# plt.tight_layout()
# plt.savefig(IMAGE_DIR / str(get_current_time()))




