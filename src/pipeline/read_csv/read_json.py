import json

# 最简单的方式
def read_json_file_simple(file_path: str) -> dict:
    """
    读取JSON文件并返回字典
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        dict: 解析后的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return {}
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return {}


# 使用示例
output_json_path = r"C:\Users\CXMO0\Documents\prediction_1hour\output\data.json"
data_dict = read_json_file_simple(output_json_path)
print(f"读取的数据类型: {type(data_dict)}")
print(f"字典的键: {list(data_dict.keys())}")

if data_dict:
    table_header = data_dict.get("table_header", [])
    time_data = data_dict.get("time_data", {})
    
    print(f"\n表头数量: {len(table_header)}")
    print(f"时间点数量: {len(time_data)}")
    
    # 查看第一个时间点
    if time_data:
        first_time = next(iter(time_data))
        print(f"第一个时间点: {first_time}")
        print(f"该时间点的数据长度: {len(time_data[first_time])}")