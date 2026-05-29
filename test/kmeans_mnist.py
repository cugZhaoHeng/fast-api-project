# 1. 导入所需库
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
IMAGE_DIR = CURRENT_DIR / 'images'

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
from utils.mnist_util import load_mnist_images, load_mnist_labels
from utils.date_util import get_current_time

# 2. 加载 MNIST 数据集（使用 fetch_openml，避免网络问题可设置 data_home）
print("正在加载 MNIST 数据集...")
X = load_mnist_images(type='train')
X = X.reshape(X.shape[0], -1)
y = load_mnist_labels(type='train')

# 为加速聚类，随机抽取 10000 个样本（可根据需要调整）
sample_size = 10000
np.random.seed(42)
indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X[indices]
y_sample = y[indices]

print(f"数据形状: {X_sample.shape}，真实标签类别数: {len(np.unique(y_sample))}")

# 3. 数据标准化（K-Means 对尺度敏感，MNIST 像素值 0~255，可归一化到 0~1）
X_sample = X_sample / 255.0

# 4. 使用 K-Means 进行聚类（n_clusters=10，因为数字 0~9）
print("正在执行 K-Means 聚类...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_sample)

# 5. 评估聚类效果（使用外部指标，需要真实标签，注意聚类标签顺序可能与真实数字不对应）
ari = adjusted_rand_score(y_sample, cluster_labels)
nmi = normalized_mutual_info_score(y_sample, cluster_labels)
print(f"调整兰德指数 (ARI): {ari:.4f}")
print(f"标准化互信息 (NMI): {nmi:.4f}")

# 6. 可视化聚类中心（每个簇中心对应一个“平均数字”）
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
centers = kmeans.cluster_centers_.reshape(10, 28, 28)

for i, ax in enumerate(axes.flat):
    ax.imshow(centers[i], cmap='gray')
    ax.set_title(f"Cluster {i}")
    ax.axis('off')

plt.suptitle("K-Means Cluster Center")
plt.tight_layout()
plt.show()

# 7. 可选：用 PCA 降维到 2D 并绘制聚类结果（展示前 2000 个点，避免过密）
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sample)

plt.figure(figsize=(12, 5))

# 左子图：按真实标签着色
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:2000, 0], X_pca[:2000, 1], c=y_sample[:2000], cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("True Cluster Label(PCA project)")

# 右子图：按聚类标签着色
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:2000, 0], X_pca[:2000, 1], c=cluster_labels[:2000], cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("K-Means Cluster Label (PCA project)")

plt.tight_layout()
plt.show()
plt.savefig(IMAGE_DIR /  f'k-means_{get_current_time()}.png')
print("图片保存成功")