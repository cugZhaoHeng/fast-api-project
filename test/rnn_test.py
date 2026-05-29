import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 获取与准备数据
# ==========================================
print("正在下载并读取天气数据...")
url = "/home/tet/zhaoheng/fast-api-project/data/weather/daily-min-temperatures.csv"
df = pd.read_csv(url, on_bad_lines='skip') # 获取气温数据
temperatures = df['Temp'].values.astype(float) # 提取出温度数值的 numpy 数组

print(f"成功获取数据，共 {len(temperatures)} 天的气温记录。")

# 神经网络对[0, 1]之间的数据更敏感，所以需要归一化 (Min-Max Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
temperatures_scaled = scaler.fit_transform(temperatures.reshape(-1, 1))

# 【核心概念：滑动窗口构造序列数据】
# RNN 不能直接看整个数组，我们需要把它切成一段段的“历史记录”和“未来目标”
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 截取过去 seq_length 天的数据作为输入特征
        X.append(data[i : i + seq_length])
        # 将紧接着的第 seq_length+1 天的数据作为预测目标
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10 # 我们用过去 10 天预测明天
X, y = create_sequences(temperatures_scaled, seq_length)

# 划分训练集 (前80%) 和测试集 (后20%)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 将 Numpy 数组转换为 PyTorch 认识的 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# 2. 定义 RNN 神经网络模型
# ==========================================
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(WeatherRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. 定义 RNN 层
        # batch_first=True 表示输入数据的维度是 [批量大小, 序列长度, 特征维度]
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 2. 定义一个全连接层 (Linear)
        # RNN 出来的隐藏状态 (记忆) 长度是 hidden_size，我们要把它变成 1 个具体的温度数值
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 将数据喂给 RNN，得到每个时间步的输出 out，和最终的隐藏状态 hn
        # x 的形状要求: [batch_size, seq_length, input_size]
        out, hn = self.rnn(x)
        
        # out 包含所有 10 天的输出，但我们只关心最后一天看完之后的“最终记忆”来预测明天
        # 取最后一个时间步的输出: out[:, -1, :] 
        last_step_out = out[:, -1, :]
        
        # 用全连接层把记忆转换为预测温度
        prediction = self.linear(last_step_out)
        return prediction

# 实例化模型: 每天只有1个特征(温度), 设定RNN隐藏记忆的维度为32
model = WeatherRNN(input_size=1, hidden_size=32, output_size=1)

# ==========================================
# 3. 训练模型
# ==========================================
criterion = nn.MSELoss() # 使用均方误差作为损失函数 (适合回归/数值预测问题)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam 优化器

epochs = 1 # 让模型把训练数据看 100 遍
print("\n开始训练...")
for epoch in range(epochs):
    model.train() # 开启训练模式
    
    # 1. 前向传播 (让网络猜一下温度)
    predictions = model(X_train)
    
    # 2. 计算误差 (猜的温度和真实温度差多少)
    loss = criterion(predictions, y_train)
    
    # 3. 反向传播与优化 (BPTT: 根据误差修正网络里的权重参数)
    optimizer.zero_grad() # 清空上一步的梯度
    loss.backward()       # 计算梯度
    optimizer.step()      # 更新参数
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], 误差 Loss: {loss.item():.4f}')

# ==========================================
# 4. 测试模型并可视化
# ==========================================
model.eval() # 开启测试模式
with torch.no_grad(): # 测试时不需要计算梯度，节约内存
    test_predictions = model(X_test)

# 因为输出是被归一化到 [0, 1] 的，我们要把它还原回真实的温度数值 (℃)
test_predictions_real = scaler.inverse_transform(test_predictions.numpy())
y_test_real = scaler.inverse_transform(y_test.numpy())

# 画出最后 200 天的预测对比图
plt.figure(figsize=(12, 5))
plt.plot(y_test_real[-200:], label='Real Temperature', color='blue', alpha=0.6)
plt.plot(test_predictions_real[-200:], label='RNN Predicted', color='red', linestyle='--')
plt.title('Melbourne Daily Minimum Temperature Prediction (RNN)')
plt.xlabel('Days')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid(True)
plt.savefig('a.png')