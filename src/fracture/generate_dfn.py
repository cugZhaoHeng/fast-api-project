import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box

# 输出目录
output_dir = r"D:\temp\diffusion\images"
os.makedirs(output_dir, exist_ok=True)

def generate_and_save_dfn(seed, output_path):
    """生成精确 256×256 像素的 DFN 图像"""
    # 参数设置
    area_size = (20, 20)  # 物理域大小 20m × 20m
    num_fractures = 30    # 适当增加裂缝数量（因分辨率提高）
    length_scale = 8       # 裂缝长度缩放
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
    
    # 设置 256×256 像素输出
    dpi = 256  # 关键参数：每英寸256点
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)  # 1英寸 × 256dpi = 256px
    
    # 绘制裂缝（调整线宽适应高分辨率）
    for line in fractures:
        x, y = line.xy
        ax.plot(x, y, color='black', linewidth=0.8)  # 线宽略微增加
    
    # 隐藏坐标轴和设置范围
    ax.set_xlim(0, area_size[0])
    ax.set_ylim(0, area_size[1])
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    # 保存图像（确保无额外填充）
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

# 生成 10 张图像（示例）
for i in range(10):
    filename = f"dfn_{i:04d}.png"
    output_path = os.path.join(output_dir, filename)
    generate_and_save_dfn(seed=i, output_path=output_path)
    print(f"Generated: {filename}")

print(f"所有图像已保存到: {output_dir}")