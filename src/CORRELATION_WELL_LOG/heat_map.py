import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lasio


import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# 1. 参数设置
# =========================
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger
logger = create_logger(__name__)
from utils.date_util import get_current_time
current_timestamp = get_current_time()
from utils.well_log_util import extract_well_data
las_name = "1055868054.las"
las_name = "1056600102.las"
las_name = "1055868076.las"
LAS_PATH = DATA_DIR / "2025_log_las" / las_name

df, _, _1 = extract_well_data(LAS_PATH)
logger.info(f"df: {df.shape}")
# 2. 预处理：删除空值过多的行/列
df = df.dropna(thresh=int(df.shape[0] * 0.5), axis=1) # 删除缺失超过 50% 的曲线
df = df.dropna(axis=0) # 删除仍含NaN的行
logger.info(f"清除异常值后的df： {df.shape}")

# 3. 计算相关性
corr = df.corr(method='pearson')

# 4. 绘图
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f"{las_name} Correlation Heatmap of Well Logs")
plt.savefig(IMAGE_DIR / f"ddpm_generate_{current_timestamp}.png")