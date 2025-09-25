from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


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

def test2(x):
    for i, v in enumerate(x):
        x[i] += 1
    return x

if __name__ == '__main__':
    my_transform = lambda x : [v + 1 for v in x]
    my_transform = lambda x : list(map(x, x))
    my_dataset = MyDataset(transform=my_transform)
    print(f"my_dataset: {my_dataset}")
    print(f"my_dataset的长度: {len(my_dataset)}")
    print(f"my_dataset.__len__(): {my_dataset[1]}")

    my_dataloader = DataLoader(my_dataset, batch_size=4, shuffle=False)

    for index, data in enumerate(my_dataloader):
        print(f"index: {index}, data: {data}")