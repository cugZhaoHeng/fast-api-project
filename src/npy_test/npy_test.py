import os
import re
import shutil
import numpy as np
from numpy import ndarray

from utils.logger import create_logger

logger = create_logger(__name__)
# 1.将3000个PERMX文件单独拷贝出来，到一个新的文件夹

def copy_permx(source_dir: str, target_dir: str, keyword: str="PERMX", suffix:str="inc") -> None:
    """
    copy file from source_dir to target_dir, only for file that contains "PERMX", and end with suffix "inc"
    :param source_dir: 源文件夹
    :param target_dir: 目标文件夹
    :return: None
    """
    logger.info(f"正在拷贝文件，从 {source_dir} 到 {target_dir} ")
    all_files = os.listdir(source_dir)
    all_files: list[str] = [f for f in all_files if f.endswith(suffix) and keyword in f]
    count: int = 1
    for filename in all_files:
        source_filepath = os.path.join(source_dir, filename)
        target_filepath = os.path.join(target_dir, filename)
        shutil.copyfile(source_filepath, target_filepath)
        count += 1
    logger.info(f"文件复制完成，共复制 {count} 个文件")

# 匹配整数、浮点数、以及类似 "数字*数字" 的表达式
pattern = r'\d+\.?\d*(?:\*\d+\.?\d*)?'
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

def save_npy(data_dir: str):
    target_files: list[str] = os.listdir(data_dir)
    for index, filename in enumerate(target_files):
        logger.info(f"文件名： {filename} ")
        current_file: str = os.path.join(target_file_dir, filename)
        value_list: list[float] = read_inc_file(current_file)
        logger.info(f"当前模型的网格数量：{len(value_list)}")
        value_array: ndarray = np.array(value_list, dtype=np.float32).reshape(16, 64, 64)
        logger.info(value_array.shape)

        order_str = ""
        if 0 <= index < 10:
            order_str = "000" + str(index)
        elif 10 <= index < 100:
            order_str = "00" + str(index)
        elif 100 <= index < 1000:
            order_str = "0" + str(index)
        elif 1000 <= index < 10000:
            order_str = str(index)

        current_filename = f"model_{order_str}.npy"
        np.save(file=os.path.join(npy_file_dir, current_filename), arr=value_array)
        logger.info(f"{current_filename}文件保存成功")



if __name__ == '__main__':
    source_file_dir: str = r"D:\Software\tnavigator22.1\demo01.snf\HM_projects\AHM_Project_1.hmf\A001\INCLUDE"
    target_file_dir: str = r"D:\temp\model_data2"
    npy_file_dir: str = r"D:\temp\npy_files"
    # copy_permx(source_file_dir, target_file_dir)

    # save_npy(target_file_dir)

    model_001 = np.load(os.path.join(npy_file_dir, "model_0001.npy"))
    model_002 = np.load(os.path.join(npy_file_dir, "model_0002.npy"))
    logger.info(f"model_001: {model_001}")
    logger.info(f"model_002: {model_002}")





