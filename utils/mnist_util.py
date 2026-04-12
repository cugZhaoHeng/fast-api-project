import gzip
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"
MNIST_DIR = DATA_DIR / "MNIST" / 'raw'

train_images_path = MNIST_DIR / "train-images-idx3-ubyte.gz"
train_labels_path = MNIST_DIR / "train-labels-idx1-ubyte.gz"
test_images_path = MNIST_DIR / "t10k-images-idx3-ubyte.gz"
test_labels_path = MNIST_DIR / "t10k-labels-idx1-ubyte.gz"

def load_mnist_images(file_path=train_images_path, type='train'):
    """
    从本地的 .gz 文件加载 MNIST 图像数据。
    参数:
        file_path: 图像文件的路径, 例如 './data/t10k-images-idx3-ubyte.gz'
    返回:
        numpy 数组, 形状为 (num_images, 28, 28)
    """
    if type == 'train':
        file_path = train_images_path
    elif type == 'test':
        file_path = test_images_path
    with gzip.open(file_path, "rb") as f:
        # 读取文件头: 魔术字, 图像数量, 行数, 列数 (均为大端整数)
        magic_number = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        # 读取剩余的图像数据并重塑形状
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)

    return data


def load_mnist_labels(file_path=train_labels_path, type='train'):
    """
    从本地的 .gz 文件加载 MNIST 标签数据。
    参数:
        file_path: 标签文件的路径, 例如 './data/t10k-labels-idx1-ubyte.gz'
    返回:
        numpy 数组, 形状为 (num_labels,)
    """
    if type == 'train':
        file_path = train_labels_path
    elif type == 'test':
        file_path = test_labels_path
    with gzip.open(file_path, "rb") as f:
        # 读取文件头: 魔术字, 标签数量 (均为大端整数)
        magic_number = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")

        # 读取剩余的标签数据
        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data

if __name__ == "__main__":
    # --- 使用示例，替换原来的 fetch_openml 部分 ---
    # 请将下面的路径替换为你的本地文件实际路径
    train_images = load_mnist_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")
    test_images = load_mnist_images(MNIST_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels(MNIST_DIR / "t10k-labels-idx1-ubyte.gz")

    print(f"训练集图像形状: {train_images.shape}")  # 预期输出: (60000, 28, 28)
    print(f"训练集标签形状: {train_labels.shape}")  # 预期输出: (60000,)
    print(f"测试集图像形状: {test_images.shape}")  # 预期输出: (10000, 28, 28)
    print(f"测试集标签形状: {test_labels.shape}")  # 预期输出: (10000,)
