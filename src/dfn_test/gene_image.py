import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 白色背景DFN生成器
class DFNGeneratorWhiteBackground:
    """离散裂缝网络生成器 - 白色背景，黑色裂缝"""

    def __init__(self, domain_size=20, resolution=64, num_fractures=10):
        self.domain_size = domain_size
        self.resolution = resolution
        self.num_fractures = num_fractures
        self.pixel_size = domain_size / resolution

    def power_law_distribution(self, a=2.0, l_min=5.0, l_max=20.0, size=1):
        """生成幂律分布的裂缝长度"""
        u = np.random.uniform(0, 1, size)
        l = l_min * (1 - u) ** (-1 / (a - 1))
        l = np.clip(l, l_min, l_max)
        return l

    def von_mises_distribution(self, mu=-30, kappa=5, size=1):
        """生成von Mises分布的裂缝方向"""
        mu_rad = np.radians(mu)
        angles = np.random.vonmises(mu_rad, kappa, size)
        angles_deg = np.degrees(angles)
        return angles_deg

    def generate_fracture_centers(self):
        """生成均匀分布的裂缝中心"""
        centers_x = np.random.uniform(0, self.domain_size, self.num_fractures)
        centers_y = np.random.uniform(0, self.domain_size, self.num_fractures)
        return centers_x, centers_y

    def generate_single_fracture_image(self, centers_x, centers_y, lengths, angles):
        """生成单个裂缝网络的图像 - 白色背景，黑色裂缝"""
        # 创建白色背景 (值为1.0)
        image = np.ones((self.resolution, self.resolution), dtype=np.float32)

        for i in range(len(centers_x)):
            half_length = lengths[i] / 2
            angle_rad = np.radians(angles[i])

            x1 = centers_x[i] - half_length * np.cos(angle_rad)
            y1 = centers_y[i] - half_length * np.sin(angle_rad)
            x2 = centers_x[i] + half_length * np.cos(angle_rad)
            y2 = centers_y[i] + half_length * np.sin(angle_rad)

            x1_pix = int(x1 / self.pixel_size)
            y1_pix = int(y1 / self.pixel_size)
            x2_pix = int(x2 / self.pixel_size)
            y2_pix = int(y2 / self.pixel_size)

            x1_pix = np.clip(x1_pix, 0, self.resolution - 1)
            y1_pix = np.clip(y1_pix, 0, self.resolution - 1)
            x2_pix = np.clip(x2_pix, 0, self.resolution - 1)
            y2_pix = np.clip(y2_pix, 0, self.resolution - 1)

            # 使用Bresenham算法绘制黑色线段
            self.draw_line(image, x1_pix, y1_pix, x2_pix, y2_pix)

        return image

    def draw_line(self, image, x0, y0, x1, y1, value=0.0):
        """使用Bresenham算法在图像上绘制线段"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0]:
                image[y0, x0] = value

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def generate_dfns(self, num_samples=1000, a=2.0, l_min=5.0, l_max=20.0, mu=-30, kappa=5):
        """生成多个DFN样本"""
        dfn_images = []

        for _ in tqdm(range(num_samples), desc="生成DFN样本(白底黑线)"):
            centers_x, centers_y = self.generate_fracture_centers()
            lengths = self.power_law_distribution(a, l_min, l_max, self.num_fractures)
            angles = self.von_mises_distribution(mu, kappa, self.num_fractures)

            image = self.generate_single_fracture_image(centers_x, centers_y, lengths, angles)
            dfn_images.append(image)

        dfn_images = np.array(dfn_images)
        return dfn_images


# 生成新的DFN数据集（白色背景，黑色裂缝）
print("生成新的DFN数据集（白色背景，黑色裂缝）...")
dfn_generator_white = DFNGeneratorWhiteBackground(domain_size=20, resolution=64, num_fractures=10)
dfn_images_white = dfn_generator_white.generate_dfns(
    num_samples=1000,
    a=2.0,
    l_min=5.0,
    l_max=20.0,
    mu=-30,
    kappa=5
)

# 可视化新生成的样本
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(dfn_images_white[i], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.tight_layout()
plt.show()

# 保存新的数据集
np.save('dfn_images_white_bg.npy', dfn_images_white)
print("新的DFN数据集已保存为 'dfn_images_white_bg.npy'")