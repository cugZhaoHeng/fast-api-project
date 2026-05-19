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
from utils.well_log_util import find_available_curves

LAS_PATH = DATA_DIR / "2025_log_las" / "1055868005.las"

# 1. 读取并转为 DataFrame
las = lasio.read(LAS_PATH)
df = las.df() 
selected = find_available_curves(df)
used_columns = list(selected.values())
df = df[used_columns].copy()

# 2. 预处理：删除空值过多的行/列
df = df.dropna(thresh=int(df.shape[0] * 0.5), axis=1) # 删除缺失超过50%的曲线
df = df.dropna(axis=0) # 删除仍含NaN的行

# 3. 计算相关性
corr = df.corr(method='pearson')

# 4. 绘图
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Well Logs")
plt.savefig(IMAGE_DIR / f"ddpm_generate_{current_timestamp}.png")