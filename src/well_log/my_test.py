from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame

csv_file_path = Path(__file__).parent / 'log.csv'
# 读取CSV文件，使用pandas库
df: DataFrame = pd.read_csv(csv_file_path)
df_head: NDFrame = df.head()
rows, cols = df.shape

df.info()
# print("df describe")
# print(df.describe())
# print(f"df_head: {df_head}")

# 找出其中的一列或多列
sp_data: DataFrame = df[['Depth', 'SP', 'GR']]
print(f"sp_data type: {type(sp_data)}")
print(f"sp_data: {sp_data[1:4]}")
# df.columns只能用来获取列名
depth_data = sp_data.columns[0]
print(f'depth_data type: {depth_data}')
# 绘制SP_DATA到数轴上，形成测井图

# fig, (axes1) = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
# axes1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# axes1.invert_yaxis()
# axes1.set_title("SP")
# axes1.set_ylabel("Depth(m)")
# curve1, = axes1.plot(sp_data['SP'], sp_data['Depth'], color='blue')
#
# axes2 = axes1.twiny()
# axes2.set_title("GR")
# curve2, = axes2.plot(sp_data['GR'], sp_data['Depth'], color='red')
#
# fig.suptitle('Well Log')
# fig.legend([curve1, curve2], ['Gamma Ray', 'Resistivity'], loc='upper left', bbox_to_anchor=(0.1, 0.95))
# plt.show()

fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
axes1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axes1.xaxis.set_label_position('top')
axes1.invert_yaxis()
axes1.set_title("SP")
axes1.set_xlabel("mV", loc='right')
axes1.set_ylabel("Depth(m)")
axes1.plot(sp_data['SP'], sp_data['Depth'], color='blue')

axes2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axes2.xaxis.set_label_position('top')
axes2.invert_yaxis()
axes2.set_title("GR")
axes2.set_xlabel("API", loc='right')
axes2.set_ylabel("Depth(m)")
axes2.plot(sp_data['GR'], sp_data['Depth'], color='red')

fig.tight_layout()
plt.show()

# sp_gr_data: DataFrame = df[['SP', 'GR']]
# print(f"sp_gr_data type: {type(sp_gr_data)}")
# print(f"sp_gr_data: {sp_gr_data}")

# # 按照索引获取列
# first_col: Series = df.loc[1]
# print(f"first_col: {first_col}")
# print(f"first_col type: {type(first_col)}")
#
# some_col: DataFrame = df.loc[:2, ['SP', 'GR']]
# print(f"some_col type: {some_col}")
# print(f"some_col type: {type(some_col)}")
#
# # 只获取一个单元格，就是标量
# one_data = df.loc[1, 'SP']
# print(f"one_data type: {type(one_data)}")
# print(f"one_data: {one_data}")

# 按照滑动窗口的大小和步长，对数据进行滑动
WINDOW_SIZE = 128
STRIDE = 32
all_data = []
for i in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    # 这里有一部分会被放弃掉，按照窗口大小来划分的话，最后的stride被放弃掉了
    windows_data = df[i:i+WINDOW_SIZE]
    all_data.append(windows_data)
d = np.stack(all_data, axis=0)
print(d.shape)
# a = [1,2,3]
# b = [2,3,4]
# c = np.stack([a], axis=-1)
# print(c)
# c = np.stack([b], axis=0)
# print(c)
# c = np.concatenate([a, b])
# print(c)
# c = np.vstack([a,b])
# print(c)
# c = np.hstack([a,b])
# print(c)
# 拿到数据之后，就拿去训练，先执行通道和列数据的交换，因为 torch 中一般使用 (B,C,L) / (B,C,W,H) 的形式
d = np.transpose(d, (0, 2, 1))
print(d.shape)

