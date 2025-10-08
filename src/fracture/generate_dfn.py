import os
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image
from shapely.geometry import LineString, box

# 输出目录
output_dir = r"images"
os.makedirs(output_dir, exist_ok=True)


def generate_and_save_dfn(seed, output_path):
    """生成精确 256×256 像素的 DFN 图像"""
    # 参数设置
    area_size = (20, 20)
    num_fractures = 10
    length_scale = 8
    length_exponent = 2.5
    angle_groups = [(0, 10), (90, 10), (60, 15)]

    # 生成 DFN 数据
    np.random.seed(seed)
    fractures = []
    width, height = area_size
    domain = box(0, 0, width, height)

    for _ in range(num_fractures):
        group_idx = np.random.randint(0, len(angle_groups))
        mean_angle, std_angle = angle_groups[group_idx]
        angle = np.random.normal(mean_angle, std_angle)
        angle_rad = np.radians(angle)
        length = length_scale * np.random.power(length_exponent)
        center_x = np.random.uniform(0, width)
        center_y = np.random.uniform(0, height)
        half_len = length / 2
        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)
        line = LineString([(center_x - dx, center_y - dy),
                           (center_x + dx, center_y + dy)])
        clipped_line = line.intersection(domain)
        if not clipped_line.is_empty and clipped_line.length > 0.2:
            fractures.append(clipped_line)

    # 创建精确的 256×256 图像
    fig = plt.figure(figsize=(256 / 100, 256 / 100), dpi=100)  # 精确控制尺寸
    ax = fig.add_axes([0, 0, 1, 1])  # 覆盖整个图形区域

    # 设置背景为白色，裂缝为黑色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 绘制裂缝 - 增加线宽以适应高分辨率
    for line in fractures:
        x, y = line.xy
        ax.plot(x, y, color='black', linewidth=3.0)  # 显著增加线宽

    # 设置坐标轴
    ax.set_xlim(0, area_size[0])
    ax.set_ylim(0, area_size[1])
    ax.set_aspect('equal')

    # 隐藏坐标轴和边框
    ax.axis('off')

    # 保存为精确的 256×256 灰度图像
    plt.savefig(output_path,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0,
                facecolor='white',
                edgecolor='none')
    plt.close()


# 替代方案：使用 NumPy 直接生成图像（更精确的控制）
def generate_dfn_numpy(seed, output_path, img_size=256):
    """使用 NumPy 直接生成 DFN 图像，确保精确尺寸"""
    area_size = (20, 20)
    num_fractures = 15
    length_scale = 8
    length_exponent = 2.5
    angle_groups = [(0, 10), (90, 10), (60, 15)]

    np.random.seed(seed)
    fractures = []
    width, height = area_size
    domain = box(0, 0, width, height)

    for _ in range(num_fractures):
        group_idx = np.random.randint(0, len(angle_groups))
        mean_angle, std_angle = angle_groups[group_idx]
        angle = np.random.normal(mean_angle, std_angle)
        angle_rad = np.radians(angle)
        length = length_scale * np.random.power(length_exponent)
        center_x = np.random.uniform(0, width)
        center_y = np.random.uniform(0, height)
        half_len = length / 2
        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)
        line = LineString([(center_x - dx, center_y - dy),
                           (center_x + dx, center_y + dy)])
        clipped_line = line.intersection(domain)
        if not clipped_line.is_empty and clipped_line.length > 0.2:
            fractures.append(clipped_line)

    # 创建空白图像（白色背景）
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255

    # 比例因子：将物理坐标转换为像素坐标
    scale_x = img_size / area_size[0]
    scale_y = img_size / area_size[1]

    # 绘制每条裂缝
    for line in fractures:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            # 转换为像素坐标
            px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
            px2, py2 = int(x2 * scale_x), int(y2 * scale_y)

            # 在图像上画线（黑色）
            rr, cc = plt.draw.line(px1, py1, px2, py2)

            # 确保坐标在图像范围内
            valid = (rr >= 0) & (rr < img_size) & (cc >= 0) & (cc < img_size)
            img[rr[valid], cc[valid]] = 0  # 设置为黑色

    # 保存图像
    Image.fromarray(img).save(output_path)


# 生成图像
print("生成 DFN 图像...")
for i in range(1000):
    filename = f"dfn_{i:04d}.png"
    output_path = os.path.join(output_dir, filename)

    # 使用方法1：matplotlib 版本
    generate_and_save_dfn(seed=i, output_path=output_path)

    # 或者使用方法2：NumPy 版本（需要安装：pip install scikit-image）
    # from skimage.draw import line as draw
    # from PIL import Image
    # generate_dfn_numpy(seed=i, output_path=output_path)

    print(f"Generated: {filename}")

print(f"所有图像已保存到: {output_dir}")