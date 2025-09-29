from typing import Optional, Callable
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# from torchvision.transforms import transforms

class MyToTensor:
    def __call__(self, input):
        return torch.tensor(input).float() + 1

class MyDataset(Dataset):
    def __init__(self, transform: Optional[Callable]=None):
        self.data = list(range(100))
        if transform is not None:
            self.transform = transform
            self.data = transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 假设你不需要标签，如果你的数据有标签，则需要返回 (image, label)
        return image

def read_images(img_dir):
    dataset = CustomImageDataset(img_dir, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    count: int = 0
    for i, images in enumerate(dataloader):
        print(f"Batch {i}:")
        print(images.shape)
        for img in images:
            count += 1
            plt.imsave(f"batch_{count}.png", arr=img.permute(1, 2, 0).numpy(), cmap='gray')
        plt.close()
        if i == 1:  # 只读取两个批次
            break

def test2(x):
    for i, v in enumerate(x):
        x[i] += 1
    return x


if __name__ == '__main__':
    # my_transform = lambda x : [v + 1 for v in x]
    # my_transform = lambda x : list(map(x, x))
    # my_dataset = MyDataset(transform=my_transform)
    # print(f"my_dataset: {my_dataset}")
    # print(f"my_dataset的长度: {len(my_dataset)}")
    # print(f"my_dataset.__len__(): {my_dataset[1]}")

    # my_dataloader = DataLoader(my_dataset, batch_size=4, shuffle=False)

    # for index, data in enumerate(my_dataloader):
    #     print(f"index: {index}, data: {data}")
    read_images('/home/tet/zhaoheng/fast-api-project/src/ldm_2d_geomodel/data/imgs1')