import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box
import geopandas as gpd
import pandas as pd

def generate_dfn(area_size=(100, 100), 
                 num_fractures=500,
                 length_scale=30, length_exponent=2.0,
                 angle_groups=[(0, 15), (90, 15)],  # (mean_angle, std_angle)
                 seed=42):
    """
    生成二维离散裂缝网络 (DFN)

    参数:
    - area_size: 研究区域大小 (width, height)
    - num_fractures: 裂缝总数
    - length_scale: 裂缝长度缩放因子（幂律分布）
    - length_exponent: 幂律分布指数（越大，小裂缝越多）
    - angle_groups: 方向组列表，每个元素为 (均值角度, 标准差)
    - seed: 随机种子

    返回:
    - GeoDataFrame 包含所有裂缝及其属性
    """
    np.random.seed(seed)
    
    # 定义研究区域边界
    width, height = area_size
    domain = box(0, 0, width, height)
    
    fractures = []
    lengths = []
    angles = []
    groups = []
    
    n_groups = len(angle_groups)
    
    for i in range(num_fractures):
        # === 1. 随机选择方向组 ===
        group_idx = np.random.randint(0, n_groups)
        mean_angle, std_angle = angle_groups[group_idx]
        
        # 随机生成角度（正态分布）
        angle = np.random.normal(mean_angle, std_angle)
        angle_rad = np.radians(angle)
        
        # === 2. 生成裂缝长度（幂律分布）===
        length = length_scale * np.random.power(length_exponent)
        
        # === 3. 随机生成中心点 ===
        center_x = np.random.uniform(0, width)
        center_y = np.random.uniform(0, height)
        
        # === 4. 计算起点和终点 ===
        half_len = length / 2
        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)
        
        start = (center_x - dx, center_y - dy)
        end = (center_x + dx, center_y + dy)
        
        # 创建线段
        line = LineString([start, end])
        
        # 可选：裁剪到研究区域内
        # line = line.intersection(domain)
        # if line.is_empty or line.length < 0.1:  # 忽略太小的片段
        #     continue
        
        # 只保留完全在区域内的线段（或你也可以用裁剪）
        if line.within(domain):
            fractures.append(line)
            lengths.append(length)
            angles.append(angle)
            groups.append(group_idx)
    
    # 构建 GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'fracture_id': range(len(fractures)),
        'length': lengths,
        'angle': angles,
        'group': groups,
        'geometry': fractures
    }, geometry='geometry')
    
    return gdf

# ======================
# 🔧 参数设置
# ======================
area_size = (100, 100)        # 区域大小 (m)
num_fractures = 300           # 裂缝数量
length_scale = 40              # 长度缩放（最大长度参考）
length_exponent = 2.5          # 幂律指数（2~3 常见）
angle_groups = [
    (0, 10),    # 近水平组
    (90, 10),   # 近垂直组
    (60, 15)    # 斜向组
]
seed = 123

# ======================
# 🚀 生成 DFN
# ======================
dfn = generate_dfn(
    area_size=area_size,
    num_fractures=num_fractures,
    length_scale=length_scale,
    length_exponent=length_exponent,
    angle_groups=angle_groups,
    seed=seed
)

print(f"成功生成 {len(dfn)} 条裂缝。")

# ======================
# 🖼️ 可视化
# ======================
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 按组上色
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, group in dfn.groupby('group'):
    group.plot(ax=ax, color=colors[i % len(colors)], linewidth=1.2, label=f'Group {i}')

# 设置图形
ax.set_xlim(0, area_size[0])
ax.set_ylim(0, area_size[1])
ax.set_aspect('equal')
ax.set_title(f'2D Discrete Fracture Network (DFN)\n'
             f'Fractures: {len(dfn)}, Area: {area_size[0]}×{area_size[1]} m²', 
             fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ======================
# 💾 可选：导出为文件
# ======================
# dfn.to_file("dfn_output.geojson", driver="GeoJSON")
# dfn.to_file("dfn_output.shp")  # Shapefile
print("可视化完成。")