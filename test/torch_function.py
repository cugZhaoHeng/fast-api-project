import torch

a = torch.tensor([1,2,3])
b = a.unsqueeze(-1)
print(b)
b = a.unsqueeze(1)
print(b)
b = a.unsqueeze(0)
print(b)

c = torch.tensor([[1,2],[3,4]])
d = c.unsqueeze(0)
print(d)
d = c.unsqueeze(1)
print(d)
d = c.unsqueeze(2)
print(d)

print(c.view(-1))
print(c.view(2, 2, 1))
print(c.view(1,4))
print(c.reshape(2,2))