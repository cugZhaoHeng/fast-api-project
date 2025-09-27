import os
import re
import shutil
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from utils.logger import create_logger

logger = create_logger(__name__)

# 匹配整数、浮点数、以及类似 "数字*数字" 的表达式
pattern = r'\d+\.?\d*(?:\*\d+\.?\d*)?'


def copy_permx(source_dir: str, target_dir: str, keyword: str = "PERMX", suffix: str = "inc") -> int:
    """
    copy file from source_dir to target_dir, only for file that contains "PERMX", and end with suffix "inc"
    :param source_dir: 源文件夹
    :param target_dir: 目标文件夹
    :param keyword: 关键词
    :param suffix: 文件后缀名
    :return: 复制成功的条数
    """

    os.makedirs(name=target_dir, exist_ok=True)
    logger.info(f"正在拷贝文件，从 {source_dir} 到 {target_dir} ")
    all_files = os.listdir(source_dir)
    all_files: list[str] = [f for f in all_files if f.endswith(suffix) and keyword in f]
    count: int = 1
    for filename in tqdm(all_files):
        source_filepath = os.path.join(source_dir, filename)
        target_filepath = os.path.join(target_dir, filename)
        shutil.copyfile(source_filepath, target_filepath)
        count += 1
    logger.info(f"文件复制完成，共复制 {count} 个文件")
    return count


def read_inc_file(filepath: str) -> list[float]:
    logger.info(f"文件路径： {filepath} ")
    with open(filepath, 'r', encoding='utf8') as f:
        value_list: list[float] = []
        while True:
            line = f.readline()
            if line == "":
                logger.info(f"文件读取完毕")
                break
            if not line.startswith(' '):
                continue
            else:
                # 从line中提取出a*b或者a的字符串，并使用一个列表接收
                matches: list[str] = re.findall(pattern, line)
                # logger.info(matches)
                for match in matches:
                    if '*' in match:
                        nums = match.split('*')
                        for i in range(int(nums[0])):
                            value_list.append(float(nums[1]))
                    else:
                        value_list.append(float(match))
    return value_list


def convert_inc_to_npy(data_dir: str, npy_dir: str) -> None:
    """
    convert inc file into npy file, and for all directory inc files
    :param data_dir: inc 文件所在的文件夹
    :param npy_dir: npy 文件保存的文件夹
    :return: None
    """
    if data_dir is None or data_dir == '':
        logger.error(f"{data_dir} 为空")
        return
    if npy_dir is None or npy_dir == '':
        logger.error(f"{npy_dir} 为空")
        return
    os.makedirs(name=npy_dir, exist_ok=True)
    target_files: list[str] = os.listdir(data_dir)
    logger.info(f"{target_files} 文件夹中文件数量 {len(target_files)}")
    for index, filename in tqdm(enumerate(target_files), total=len(target_files), desc="处理文件"):
        current_file: str = os.path.join(target_file_dir, filename)
        value_list: list[float] = read_inc_file(current_file)
        value_array: np.ndarray = np.array(value_list, dtype=np.float32).reshape(16, 64, 64)
        logger.info(value_array.shape)

        # 简化编号格式化（更优雅的写法）
        order_str = f"{index:04d}"  # 自动补零到4位，等价于之前的 if-elif 结构

        current_filename = f"model_{order_str}.npy"
        np.save(file=os.path.join(npy_dir, current_filename), arr=value_array)


if __name__ == '__main__':
    source_file_dir: str = r"E:\INCLUDE"
    target_file_dir: str = r"E:\model_data2"
    npy_file_dir: str = r"../data/npy_files"
    # copy_permx(source_file_dir, target_file_dir)

    convert_inc_to_npy(data_dir=target_file_dir, npy_dir=npy_file_dir)
    # save_npy(target_file_dir)

    # model_001 = np.load(os.path.join(npy_file_dir, "model_0001.npy"))
    # model_002 = np.load(os.path.join(npy_file_dir, "model_0002.npy"))
    # logger.info(f"model_001: {model_001}")
    # logger.info(f"model_002: {model_002}")
