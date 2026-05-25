# 专门用来手写一个MNIST的识别，精度不管
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# 读取数据，分为训练集和测试集
import sys
from pathlib import Path
# 将项目根目录添加到系统路径。对应修改 parent level
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(project_root)
DATA_DIR = project_root / "data"
my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])
train_datasets = datasets.MNIST(root=DATA_DIR, train=True, transform=my_transform, download=False)
train_dataloader = DataLoader(dataset=train_datasets, batch_size=128, shuffle=True)
test_datasets = datasets.MNIST(root=DATA_DIR, train=False, transform=my_transform, download=False)
test_dataloader = DataLoader(dataset=test_datasets, batch_size=128, shuffle=False)

# 编写训练过程

class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
        
    
    def forward(self, x):
        # 先将图像展平
        x = x.view(x.size(0), -1)
        a = self.mlp(x)
        return a

device = torch.device("cuda:0")

def train():
    model = MnistClassifier()
    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=0.001)
    model.train()
    loss_function =nn.MSELoss()
    for x, y in train_dataloader:
        # print(f"x: {x}")
        # print(f"y shape: {y.shape}: {y}")
        x = x.to(device)
        y = y.to(device)
        # 转化为 one-hot 编码
        y_onehot = F.one_hot(y, num_classes=10).float()
        # print(f"y_onehot shape: {y_onehot.shape}")
        optimizer.zero_grad()
        y_pred  = model.forward(x)
        # print(f"y_pred shape:{y_pred.shape}: {y_pred}")
        
        loss = loss_function(y_onehot, y_pred)
        loss.backward()
        optimizer.step()
        # print(f"loss: {loss.item()}")
    print("训练完成")
    
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_dataloader))
        x = x.to(device)
        y = y.to(device)
        y_pred = model.forward(x)
        y_pred = torch.argmax(y_pred, dim=1)
        print(y_pred)
        

if __name__ == "__main__":
    train()
    # x, y = next(iter(test_dataloader))
    # x = x.to(device)
    # y = y.to(device)
    # print(x)
    # print(y)