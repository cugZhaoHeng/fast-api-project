# 张量的形状操作
import torch

x = torch.arange(36)

x = x.reshape(3,12)
x = x.reshape(-1)
x = x.view(-1, 4)
print(x)

# y = torch.range(1, 10, step=1)
# print(y)


