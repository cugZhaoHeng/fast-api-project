import os
import lasio
import pandas as pd

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
    "RLL3": ["RLL3", "LLS", "LL3", "RES_SHALLOW"]
}

def extract_well_data(file_path):
    """
    读取文件，并使用你的逻辑提取数据，同时处理别名映射
    """
    las = lasio.read(file_path)
    df = las.df()
    
    final_data = pd.DataFrame(index=df.index)
    mapping_log = {} # 记录这口井具体用了哪一列

    for target_name, candidates in ALIASES.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        
        if found:
            final_data[target_name] = df[found]
            mapping_log[target_name] = found
        else:
            return None, None # 只要少一列，这口井就作废

    return final_data, mapping_log

# 主处理逻辑
def process_all_wells(folder_path):
    all_wells_data = {}
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".las"):
            file_path = os.path.join(folder_path, filename)
            
            data, mapping = extract_well_data(file_path)
            
            if data is not None:
                all_wells_data[filename] = data
                print(f"[成功] 提取井: {filename}, 映射为: {mapping}")
            else:
                print(f"[跳过] 井 {filename} 缺少必要曲线")
                
    return all_wells_data

# 使用方式
wells_data = process_all_wells("/home/tet/zhaoheng/fast-api-project/data/2025_log_las")
# 现在 wells_data['井名.las'] 就是一个标准化的 DataFrame，列名全是 ['GR', 'SP'...]
print(wells_data)