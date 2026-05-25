import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from PIL import ImageFilter

# --- 参数配置 ---
NUM_SAMPLES = 10000          # 生成样本的总数
IMAGE_SIZE = 128            # 图像分辨率提升至 128x128
L_MIN = 20.0
L_MAX = 80.0

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
# 数据保存路径
OUTPUT_DIR = DATA_DIR / "dfn_data_1_20"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)

# 设置随机种子以保证结果可复现
np.random.seed(42)

# --- 1. 统计分布生成函数 ---

def sample_power_law(a, l_min, l_max, size=1):
    """
    使用逆变换法采样幂律分布的裂缝长度 l ~ l^(-a)
    """
    u = np.random.uniform(0.0, 1.0, size)
    temp = u * (l_max**(1.0 - a) - l_min**(1.0 - a)) + l_min**(1.0 - a)
    lengths = temp**(1.0 / (1.0 - a))
    return lengths

def sample_von_mises(mu_deg, kappa, size=1):
    """
    采样冯·米塞斯分布的走向角度
    """
    mu_rad = np.radians(mu_deg)
    angles_rad = np.random.vonmises(mu_rad, kappa, size)
    return angles_rad

# --- 2. 核心生成逻辑 ---

logger_data = []

print(f"开始生成 {NUM_SAMPLES} 个 DFN 样本...")
for idx in tqdm(range(NUM_SAMPLES)):
    # 2.1 随机采样该张图的全局控制参数（条件标签）
    a = np.random.uniform(1.5, 3.0)               # 幂律指数
    mu_deg = np.random.uniform(-90.0, 90.0)       # 平均走向（度）
    kappa = np.random.uniform(1.0, 10.0)          # 集中度
    num_fractures = int(np.random.randint(1, 20))  # 裂缝条数
    
    # 2.2 创建白色背景画布 (255 代表白色背景)
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    draw = ImageDraw.Draw(img)
    
    # 2.3 生成单张图中的每一条裂缝
    lengths = sample_power_law(a, L_MIN, L_MAX, size=num_fractures)
    angles = sample_von_mises(mu_deg, kappa, size=num_fractures)
    
    # 随机均匀分布裂缝中心位置 (X, Y)
    centers_x = np.random.uniform(0, IMAGE_SIZE, size=num_fractures)
    centers_y = np.random.uniform(0, IMAGE_SIZE, size=num_fractures)
    
    for i in range(num_fractures):
        cx, cy = centers_x[i], centers_y[i]
        length = lengths[i]
        theta = angles[i]
        
        # 计算裂缝的两个端点坐标
        dx = (length / 2.0) * math.cos(theta)
        dy = (length / 2.0) * math.sin(theta)
        
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy
        
        # 在画布上绘制线宽为 1 像素的黑线 (0 代表黑色)
        line_width = np.random.randint(2, 5)

        draw.line(
            (x1, y1, x2, y2),
            fill=0,
            width=line_width
        )
        
    # 2.4 保存图片
    img_name = f"dfn_{idx:05d}.png"
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img.save(img_path)
    
    # 2.5 记录该样本的条件标签
    logger_data.append({
        "image_name": img_name,
        "exponent_a": round(a, 4),
        "mean_mu": round(mu_deg, 4),
        "concentration_kappa": round(kappa, 4),
        "num_fractures": num_fractures
    })

# --- 3. 保存标签至 CSV 文件 ---
df = pd.DataFrame(logger_data)
df.to_csv(CSV_PATH, index=False)

print("\n数据生成完毕！")