
import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def process_csv_files(folder_path: str) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    处理文件夹下的所有CSV文件
    
    Args:
        folder_path: CSV文件所在的文件夹路径
        
    Returns:
        Tuple[List[str], Dict[str, List[float]]]: 
            - 表头列表（不包含TIME）
            - 字典：key为时间，value为对应时间所有数据的列表
    """
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"在文件夹 {folder_path} 中未找到CSV文件")
    
    # 存储所有表头（不包含TIME）
    all_headers = []
    # 存储数据：key为时间，value为数据列表
    data_dict = defaultdict(list)
    # 记录每个文件的列顺序
    file_column_orders = []
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 第一次遍历：收集所有表头并初始化数据结构
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # 读取表头
                
                # 验证第一列是否为TIME
                if headers[0] != 'TIME':
                    print(f"警告: 文件 {csv_file} 的第一列不是'TIME'，而是'{headers[0]}'")
                
                # 记录该文件的列顺序（不包含TIME列）
                non_time_headers = headers[1:]
                file_column_orders.append(non_time_headers)
                all_headers.extend(non_time_headers)
                
                # 读取第一行数据以确定时间格式和初始化字典
                first_row = next(reader)
                time_key = first_row[0]
                
                # 初始化该时间对应的列表（用None占位）
                for time_key in [row[0] for row in reader]:
                    data_dict[time_key] = [None] * len(all_headers)
                    
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
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                
                for row in reader:
                    if not row:  # 跳过空行
                        continue
                        
                    time_key = row[0]
                    
                    # 确保该时间存在于字典中
                    if time_key not in data_dict:
                        data_dict[time_key] = [None] * len(all_headers)
                    
                    # 填充该时间对应的数据
                    for col_idx, value in enumerate(row[1:]):
                        # 计算在总列表中的位置
                        total_index = current_header_index + col_idx
                        
                        # 转换值为浮点数
                        try:
                            float_value = float(value)
                        except ValueError:
                            float_value = 0.0  # 如果转换失败，使用0.0
                        
                        data_dict[time_key][total_index] = float_value
        
            # 更新当前表头索引位置
            current_header_index += len(column_headers)
            print(f"已处理文件 {csv_file}，添加了 {len(column_headers)} 列")
            
        except Exception as e:
            print(f"读取文件 {csv_file} 数据时出错: {e}")
            continue
    
    # 验证数据完整性
    incomplete_times = [time for time, values in data_dict.items() if None in values]
    if incomplete_times:
        print(f"警告: {len(incomplete_times)} 个时间点的数据不完整")
    
    return all_headers, dict(data_dict)


def save_to_files(all_headers: List[str], data_dict: Dict[str, List[float]], 
                  output_folder: str = "output"):
    """
    将处理结果保存到文件
    
    Args:
        all_headers: 表头列表
        data_dict: 数据字典
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. 保存表头到文件
    headers_file = os.path.join(output_folder, "headers.txt")
    with open(headers_file, 'w', encoding='utf-8') as f:
        f.write("所有表头（不包含TIME）:\n")
        for i, header in enumerate(all_headers):
            f.write(f"{i+1}. {header}\n")
    
    # 2. 保存数据到CSV文件
    data_file = os.path.join(output_folder, "combined_data.csv")
    with open(data_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['TIME'] + all_headers)
        
        # 写入数据
        for time_key, values in sorted(data_dict.items()):
            writer.writerow([time_key] + values)
    
    # 3. 保存统计数据
    stats_file = os.path.join(output_folder, "statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据统计信息:\n")
        f.write(f"表头总数: {len(all_headers)}\n")
        f.write(f"时间点数: {len(data_dict)}\n")
        f.write(f"数据总数: {len(all_headers) * len(data_dict)}\n")
        
        # 计算非零值比例
        total_values = len(all_headers) * len(data_dict)
        zero_count = 0
        for values in data_dict.values():
            zero_count += sum(1 for v in values if v == 0.0)
        
        if total_values > 0:
            zero_percentage = (zero_count / total_values) * 100
            f.write(f"零值比例: {zero_percentage:.2f}%\n")
    
    print(f"结果已保存到 {output_folder} 文件夹")
    print(f"- 表头文件: {headers_file}")
    print(f"- 合并数据文件: {data_file}")
    print(f"- 统计文件: {stats_file}")


def display_sample_data(all_headers: List[str], data_dict: Dict[str, List[float]], 
                       num_samples: int = 3):
    """
    显示示例数据
    
    Args:
        all_headers: 表头列表
        data_dict: 数据字典
        num_samples: 显示的时间点数量
    """
    print("\n" + "="*80)
    print("示例数据:")
    print("="*80)
    
    print(f"前 {len(all_headers)} 个表头: {all_headers[:5]}...")
    print(f"总表头数: {len(all_headers)}")
    print(f"总时间点数: {len(data_dict)}")
    print("\n前几个时间点的数据:")
    
    sample_count = 0
    for time_key, values in data_dict.items():
        if sample_count >= num_samples:
            break
        
        print(f"\n时间: {time_key}")
        print(f"数据长度: {len(values)}")
        
        # 显示前几个值
        if len(values) > 0:
            print(f"前10个值: {values[:10]}")
        
        sample_count += 1


def main():
    # 配置参数
    folder_path = r"C:\Users\CXMO0\Documents\prediction_1hour"  # 请修改为实际的文件夹路径
    output_folder = r"C:\Users\CXMO0\Documents\output"
    
    try:
        # 处理CSV文件
        all_headers, data_dict = process_csv_files(folder_path)
        
        # 显示示例数据
        display_sample_data(all_headers, data_dict)
        
        # 保存结果到文件
        save_to_files(all_headers, data_dict, output_folder)
        
        print("\n处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 使用示例
if __name__ == "__main__":
    main()