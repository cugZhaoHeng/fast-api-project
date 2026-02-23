import os
import csv
import json
from collections import defaultdict
from typing import Dict, List, Any


def process_csv_files_to_dict(folder_path: str) -> Dict[str, Any]:
    """
    处理文件夹下的所有CSV文件，返回指定格式的字典

    Args:
        folder_path: CSV文件所在的文件夹路径

    Returns:
        Dict[str, Any]: 包含table_header和time_data的字典
    """
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'predictions' in f]

    if not csv_files:
        raise ValueError(f"在文件夹 {folder_path} 中未找到CSV文件")

    # 存储所有表头（不包含TIME）
    all_headers = []
    # 存储数据：key为时间，value为数据列表
    time_data_dict = {}
    # 记录每个文件的列顺序
    file_column_orders = []

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 第一次遍历：收集所有表头并初始化数据结构
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t' if '\t' in f.read(1024) else ',')
                f.seek(0)  # 重置文件指针
                headers = next(reader)  # 读取表头

                # 验证第一列是否为TIME
                if headers[0] != 'TIME':
                    print(f"注意: 文件 {csv_file} 的第一列不是'TIME'，而是'{headers[0]}'，将使用第一列作为时间列")

                # 记录该文件的列顺序（不包含时间列）
                non_time_headers = headers[1:]
                file_column_orders.append(non_time_headers)
                all_headers.extend(non_time_headers)

                # 读取数据以初始化字典
                for row in reader:
                    if not row:  # 跳过空行
                        continue
                    time_key = row[0]
                    if time_key not in time_data_dict:
                        # 初始化该时间对应的列表（用None占位）
                        time_data_dict[time_key] = [None] * len(all_headers)

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            continue

    print(f"总共收集到 {len(all_headers)} 个表头")

    # 第二次遍历：填充数据
    current_header_index = 0

    for file_idx, (csv_file, column_headers) in enumerate(zip(csv_files, file_column_orders)):
        file_path = os.path.join(folder_path, csv_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 自动检测分隔符
                sample = f.read(1024)
                f.seek(0)
                delimiter = '\t' if '\t' in sample else ','

                reader = csv.reader(f, delimiter=delimiter)
                next(reader)  # 跳过表头

                for row in reader:
                    if not row:  # 跳过空行
                        continue

                    time_key = row[0]

                    # 确保该时间存在于字典中
                    if time_key not in time_data_dict:
                        time_data_dict[time_key] = [None] * len(all_headers)

                    # 填充该时间对应的数据
                    for col_idx, value in enumerate(row[1:]):
                        # 计算在总列表中的位置
                        total_index = current_header_index + col_idx

                        # 转换值为浮点数
                        try:
                            # 处理空字符串或异常值
                            if value == '' or value.lower() == 'null' or value.lower() == 'none':
                                float_value = None
                            else:
                                float_value = float(value)
                        except (ValueError, TypeError):
                            float_value = None  # 如果转换失败，使用None

                        time_data_dict[time_key][total_index] = float_value

            # 更新当前表头索引位置
            current_header_index += len(column_headers)
            print(f"已处理文件 {csv_file}，添加了 {len(column_headers)} 列")

        except Exception as e:
            print(f"读取文件 {csv_file} 数据时出错: {e}")
            continue

    # 构建最终输出字典
    result_dict = {
        "inputKey": all_headers,
        "inputValue": time_data_dict
    }

    return result_dict


def save_to_json(data_dict: Dict[str, Any], output_path: str = "output.json", 
                 indent: int = 2, ensure_ascii: bool = False):
    """
    将字典保存为JSON文件
    
    Args:
        data_dict: 要保存的字典
        output_path: 输出JSON文件路径
        indent: JSON缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=indent, ensure_ascii=ensure_ascii)
        print(f"数据已成功保存到 {output_path}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")


def display_summary(data_dict: Dict[str, Any], show_samples: int = 3):
    """
    显示数据摘要信息
    
    Args:
        data_dict: 数据字典
        show_samples: 显示的样本数量
    """
    table_header = data_dict.get("table_header", [])
    time_data = data_dict.get("time_data", {})
    
    print("\n" + "="*80)
    print("数据摘要信息:")
    print("="*80)
    print(f"表头数量: {len(table_header)}")
    print(f"时间点数量: {len(time_data)}")
    
    if table_header:
        print(f"\n前10个表头: {table_header[:10]}")
        if len(table_header) > 10:
            print(f"... (共 {len(table_header)} 个)")
    
    if time_data:
        print(f"\n前 {show_samples} 个时间点的数据:")
        time_keys = list(time_data.keys())[:show_samples]
        for time_key in time_keys:
            values = time_data[time_key]
            # 统计None值
            none_count = sum(1 for v in values if v is None)
            print(f"  时间: {time_key}")
            print(f"    数据长度: {len(values)}")
            print(f"    None值数量: {none_count}")
            # 显示前几个非None值
            non_none_values = [v for v in values if v is not None]
            if non_none_values:
                print(f"    前5个非None值: {non_none_values[:5]}")
            else:
                print(f"    所有值均为None")


def validate_data(data_dict: Dict[str, Any]) -> bool:
    """
    验证数据完整性
    
    Args:
        data_dict: 数据字典
        
    Returns:
        bool: 数据是否完整
    """
    table_header = data_dict.get("table_header", [])
    time_data = data_dict.get("time_data", {})
    
    if not table_header:
        print("警告: table_header为空!")
        return False
    
    if not time_data:
        print("警告: time_data为空!")
        return False
    
    # 检查所有时间点的数据长度是否与表头长度一致
    header_len = len(table_header)
    inconsistent_times = []
    
    for time_key, values in time_data.items():
        if len(values) != header_len:
            inconsistent_times.append((time_key, len(values)))
    
    if inconsistent_times:
        print(f"警告: 有 {len(inconsistent_times)} 个时间点的数据长度与表头长度不一致")
        for time_key, actual_len in inconsistent_times[:5]:  # 只显示前5个
            print(f"  时间 {time_key}: 表头长度={header_len}, 实际数据长度={actual_len}")
        if len(inconsistent_times) > 5:
            print(f"  ... (共 {len(inconsistent_times)} 个)")
    
    return True


def main():
    """
    主函数
    """
    # 配置参数
    folder_path = r"D:\git\fast-api-project\src\pipeline\data\processed_files"  # 请修改为实际的文件夹路径
    output_json_path =  r"D:\git\fast-api-project\src\pipeline\data\data.json"  # 输出JSON文件路径
    
    try:
        # 处理CSV文件，获取字典
        print("开始处理CSV文件...")
        result_dict = process_csv_files_to_dict(folder_path)
        
        # 验证数据完整性
        if validate_data(result_dict):
            print("数据验证通过!")
        else:
            print("数据验证失败，但将继续处理...")
        
        # 显示摘要信息
        display_summary(result_dict)
        
        # 保存为JSON文件
        save_to_json(result_dict, output_json_path)
        
        # 也可以打印一小部分JSON内容供预览
        print(f"\nJSON文件预览（前200个字符）:")
        json_str = json.dumps(result_dict, ensure_ascii=False, indent=2)
        print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 快速使用示例
def quick_process():
    """快速处理示例"""
    folder_path = "你的csv文件夹路径"
    
    try:
        # 处理CSV文件
        result_dict = process_csv_files_to_dict(folder_path)
        
        # 显示基本信息
        headers = result_dict["table_header"]
        time_data = result_dict["time_data"]
        
        print(f"表头总数: {len(headers)}")
        print(f"时间点数量: {len(time_data)}")
        
        # 获取第一个时间点
        first_time = next(iter(time_data))
        print(f"\n第一个时间点 {first_time}:")
        print(f"  数据长度: {len(time_data[first_time])}")
        
        # 保存为JSON
        with open("quick_output.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据已保存到 quick_output.json")
        
    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == "__main__":
    # 运行主函数
    main()
    
    # 或者使用快速处理
    # quick_process()