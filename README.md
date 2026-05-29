当前分支，是我用来编写 扩散模型和地质模型 的相关代码的分支，包含 MNIST， VAE， Diffusion 等的练习

每个项目或者文件中都需要使用的代码，用来设置项目路径、当前目录、图片保存路径、模型保存路径
```python
import os
from pathlib import Path
from utils.logger import create_logger

logger = create_logger(__name__)
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'
MNIST_DIR = PROJECT_ROOT_DIR / 'data' / 'MNIST'
IMAGE_DIR = CURRENT_DIR / 'images'
MODEL_DIR = CURRENT_DIR / 'models'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pth"
```

在训练模型之前，先读取是否有训练了一半的模型，如果有就加载，没有的话就直接从头开始训练


依然需要遵循如下规则：
1. 项目分为 src 源代码和 util 工具类， 以及 data 公共数据， 公共数据主要包含 MNIST 的data，日志统一写在一个目录下
2. 每个文件夹下有自己的 images 目录和 models 目录，用来存放代码在运行过程中产生的图片和模型
3. images 和 models 目录不上传，因为体积太大了
4. 在编写路径的时候，文件夹统一添加后缀 _DIR, 文件路径统一添加后缀 _PATH， 用来区分，函数参数也要遵守
5. 代码中不再使用 print 打印信息，统一使用 logger.info来打印日志
6. 每个训练代码中，都需要先读取最近的模型，以免每次都从头开始训练
7. 模型的保存使用统一的规则，即模型有哪些键，读取的规则也设定成一样的
8. 在出图的时候，不要出现中文，统一使用英文，以免出现乱码
9. 要区分维度和形状的概念
10. todo ok