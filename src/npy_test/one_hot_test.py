import numpy as np

# 这里的目的是，将当前的数字，转化成为one-hot编码，也就是用一维矩阵来表示，某个位置是1，其他位置是0
a = [0.1, 10, 200]


a = [[0,1,1,2],[0,1,1,2]]
a = np.array(a)
b = np.eye(3)
print(b)
c = np.eye(3)[a] # 这行代码的含义是，将a里面的元素，按照当前的单位对角线矩阵
print(c)