import os
from pathlib import Path
import shutil
import sys
import lasio
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"

project_root_str = str(PROJECT_ROOT_DIR)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from utils.logger import create_logger

logger = create_logger(__name__)
from utils.date_util import get_current_time

current_timestamp = get_current_time()

LAS_DIR = DATA_DIR / "2025_log_las"
LAS_PATH = DATA_DIR / "2025_log_las" / "1055868005.las"

# 你的别名配置（优先级：列表越靠前，优先级越高）
ALIASES = {
    "GR": ["GR", "GR_UNIT", "SGR"],
    "SP": ["SP"],
    "DCAL": ["DCAL", "CAL", "CALI", "HCAL"],
    "RHOB": ["RHOB", "DEN", "DENSITY"],
    "DT": ["DT", "AC", "SONIC"],
    "CNPOR": ["CNPOR", "NPOR", "TNPH"],
    "RILD": ["RILD", "ILD", "RT", "RES_DEEP"],
    "RILM": ["RILM", "ILM", "RES_MEDIUM"],
    "RLL3": ["RLL3", "LLS", "LL3", "RES_SHALLOW"],
}


def extract_well_data(file_path):
    """
    读取文件，标准化列名，并将深度设为Index
    """
    try:
        las = lasio.read(file_path)
        df = las.df()
    except Exception as e:
        return None, None, f"读取文件错误: {e}"

    # 1. 确保深度被设置为 Index
    # lasio.read().df() 通常会自动把深度作为 index，这里做个强制保障
    if df.index.name not in ['DEPT', 'DEPTH', 'TVD']:
        logger.info(f"没有找到深度列")
        # 如果 index 没设好，尝试在 columns 里找深度列
        depth_cols = [c for c in df.columns if c.lower() in ['dept', 'depth', 'tvd']]
        if depth_cols:
            df = df.set_index(depth_cols[0])
    
    final_data = pd.DataFrame(index=df.index)
    mapping_log = {}

    # 2. 提取并标准化列名
    for target_name, candidates in ALIASES.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        
        if found:
            # 检查是否有数据
            if df[found].isnull().all():
                logger.info(f"存在空数据")
                return None, None, f"曲线 {target_name} ({found}) 数据全为空"

            # 将提取出的数据列名统一为 ALIASES 的 key (即 GR, SP 等)
            final_data[target_name] = df[found]
            mapping_log[target_name] = found
        else:
            logger.info(f"缺少必要的曲线")
            return None, None, f"缺少必要曲线: {target_name}"

    # 返回数据：final_data 的 index 是深度，columns 是 ['GR', 'SP', ..., 'RLL3']
    return final_data, mapping_log, "成功"


def process_and_copy_wells(source_folder, target_folder):
    """处理并拷贝文件"""
    # 确保目标文件夹存在
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    all_wells_data = {}

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".las"):
            file_path = os.path.join(source_folder, filename)

            try:
                # 核心：将所有逻辑包裹在 try 中，防止单文件异常终止程序
                data, mapping, status = extract_well_data(file_path)

                if data is not None:
                    # 1. 存入内存数据字典
                    all_wells_data[filename] = data

                    # 2. 拷贝文件到目标目录
                    dest_path = os.path.join(target_folder, filename)
                    shutil.copy2(file_path, dest_path)

                    print(f"[成功] 提取并拷贝: {filename}")
                else:
                    print(f"[跳过] {filename} -> {status}")

            except Exception as e:
                # 捕获任何意料之外的错误（如权限、磁盘满等）
                print(f"[异常] 处理 {filename} 时发生意外: {e}")
                continue  # 继续处理下一口井

    return all_wells_data


if __name__ == "__main__":
    # source_dir = r"C:\Users\Administrator\Downloads\2025"
    # dest_dir = r"C:\Users\Administrator\Downloads\Qualified_Wells"
    # wells_data = process_and_copy_wells(source_dir, dest_dir)
    data, log, _ = extract_well_data(LAS_DIR / "1055868005.las")
