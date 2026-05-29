import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CustomGrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # 生成 image_0.jpg 到 image_4999.jpg 的文件名列表
        self.image_names = [f"image_{i}.jpeg" for i in range(5000)]
        # 过滤掉不存在的文件
        self.image_names = [n for n in self.image_names if (self.root_dir / n).exists()]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.image_names[idx]
        # 读取并转为灰度图 ('L')
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        # 返回 (图片, 占位标签)，占位标签设为 0
        return image, 0

def get_vae_dataloaders(data_dir, transform, batch_size=128, split_ratio=0.9):
    """
    封装数据加载器
    返回: (train_dataset, train_loader, test_loader)
    """
    full_dataset = CustomGrayDataset(data_dir, transform)
    
    # 划分训练集和测试集
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_ds, train_loader, test_loader