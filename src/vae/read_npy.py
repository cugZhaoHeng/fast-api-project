import numpy as np

# 加载数据
a = np.load("./generated_F/random_facies_model_01.npy")
b = a.reshape(-1)  # 展平成一维数组

# 用空格分隔写入文本文件
np.savetxt("a_F.txt", b, fmt='%.6g', delimiter=' ')