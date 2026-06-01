import torch

x = torch.tensor([1,2])
y = torch.tensor([3,4])

c = torch.concat((x, y), dim=0)
print(f"c: {c}")
# c = torch.concat((x, y), dim=1)
print(f"c: {c}")

c = torch.stack([x, y])
print(f"c: {c}")

c = torch.add(x, y)
print(f"c: {c}")


# 测试多维度张量的函数
x = torch.tensor([[1,2,], [3,4]])
y = torch.tensor([[5,6,], [7,8]])
# concat 函数
c = torch.concat((x, y))
print(f"c: {c}")

c = torch.concat(tensors=(x, y), dim=1)
print(f"c: {c}")
# stack函数
c = torch.stack(tensors = [x, y], dim=0)
print(f"c: {c}")

c = torch.stack(tensors = [x, y], dim=1)
print(f"c: {c}")
# add函数
c = torch.add(x, y)
print(f"c: {c}")

c = torch.cat((x, y))
print(f"c: {c}")

c = x + y
print(f"c: {c}")
