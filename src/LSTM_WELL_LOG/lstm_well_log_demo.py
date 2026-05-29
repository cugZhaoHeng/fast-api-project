# lstm_well_log_demo.py
# pip install numpy pandas matplotlib scikit-learn torch lasio

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

LAS_PATH = DATA_DIR / "2025_log_las" / "1056600514.las"

INPUT_CURVES = ["GR", "RHOB", "NPHI", "ILD"]  
TARGET_CURVE = "GR"

WINDOW_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 读取 LAS 数据
# =========================

def load_las_as_dataframe(las_path):
    las = lasio.read(las_path)
    df = las.df().reset_index()

    # 第一列通常是 DEPT / DEPTH
    df.rename(columns={df.columns[0]: "DEPTH"}, inplace=True)

    print("LAS 中包含的曲线：")
    print(df.columns.tolist())

    return df


def find_available_curves(df, curve_names):
    """
    有些 LAS 文件里电阻率可能叫 ILD、RILD、LLD、RT 等。
    这里做一个简单兼容。
    """
    aliases = {
        "GR": ["GR", "GAM", "CGR"],
        "RHOB": ["RHOB", "RHOZ", "DEN"],
        "NPHI": ["NPHI", "NPOR", "CNPOR", "CNLS"],
        "ILD": ["ILD", "RILD", "LLD", "RT", "RES"]
    }

    selected = {}

    for name in curve_names:
        candidates = aliases.get(name, [name])
        found = None

        for c in candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            raise ValueError(f"没有找到曲线 {name}，可选列为：{df.columns.tolist()}")

        selected[name] = found

    return selected


# =========================
# 3. 构造数据集
# =========================

class WellLogDataset(Dataset):
    def __init__(self, x, y, window_size):
        self.x = x
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.x) - self.window_size

    def __getitem__(self, idx):
        x_window = self.x[idx: idx + self.window_size]
        y_next = self.y[idx + self.window_size]

        return (
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y_next, dtype=torch.float32)
        )


# =========================
# 4. 定义 LSTM 模型
# =========================

class WellLogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的隐藏状态
        last_hidden = out[:, -1, :]

        y_pred = self.fc(last_hidden)

        return y_pred.squeeze(-1)


# =========================
# 5. 训练函数
# =========================

def train_model(model, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss: {avg_train_loss:.6f} "
                f"Test Loss: {avg_test_loss:.6f}"
            )

    return train_losses, test_losses


# =========================
# 6. 测试与画图
# =========================

def evaluate_and_plot(model, test_dataset, y_scaler):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]

            x = x.unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().item()

            preds.append(pred)
            trues.append(y.item())

    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)

    # 反标准化
    preds_inv = y_scaler.inverse_transform(preds).flatten()
    trues_inv = y_scaler.inverse_transform(trues).flatten()

    mse = mean_squared_error(trues_inv, preds_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_inv, preds_inv)
    r2 = r2_score(trues_inv, preds_inv)

    print("\n测试集结果：")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R²   = {r2:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(trues_inv, label="True GR")
    plt.plot(preds_inv, label="Predicted GR")
    plt.xlabel("Test sample index")
    plt.ylabel("GR")
    plt.title("LSTM Well Log Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "predict.png")


def plot_loss(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "loss.png")


# =========================
# 7. 主程序
# =========================

def main():
    df = load_las_as_dataframe(LAS_PATH)

    selected = find_available_curves(df, INPUT_CURVES)
    print("\n实际使用的曲线映射：")
    print(selected)

    used_columns = ["DEPTH"] + list(selected.values())

    df = df[used_columns].copy()

    # 去掉缺失值，保证当前示例使用完整数据
    df = df.replace([-999.25, -999.0], np.nan)
    df = df.dropna().reset_index(drop=True)

    print("\n清洗后的数据量：", len(df))
    print(df.head())

    feature_cols = list(selected.values())
    target_col = selected[TARGET_CURVE]
    logger.info(f"feature_cols: {feature_cols}")
    logger.info(f"target_col: {target_col}")

    x_raw = df[feature_cols].values
    y_raw = df[[target_col]].values
    logger.info(f"x_raw: {x_raw}")
    logger.info(f"y_raw: {y_raw}")
    
    # 标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(x_raw)
    y_scaled = y_scaler.fit_transform(y_raw).flatten()

    # 按深度顺序划分，前 80% 训练，后 20% 测试
    split_idx = int(len(df) * 0.95)

    x_train = x_scaled[:split_idx]
    y_train = y_scaled[:split_idx]

    x_test = x_scaled[split_idx - WINDOW_SIZE:]
    y_test = y_scaled[split_idx - WINDOW_SIZE:]

    train_dataset = WellLogDataset(x_train, y_train, WINDOW_SIZE)
    test_dataset = WellLogDataset(x_test, y_test, WINDOW_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = WellLogLSTM(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        output_size=1
    ).to(DEVICE)

    print("\n模型结构：")
    print(model)

    train_losses, test_losses = train_model(
        model,
        train_loader,
        test_loader
    )

    plot_loss(train_losses, test_losses)

    evaluate_and_plot(
        model,
        test_dataset,
        y_scaler
    )


if __name__ == "__main__":
    main()