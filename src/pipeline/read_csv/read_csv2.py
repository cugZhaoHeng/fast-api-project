import os
import csv
import json
import glob
from collections import defaultdict


def process_csv_files_correctly(folder_path: str):
    """
    正确处理CSV文件的函数
    
    Args:
        folder_path: CSV文件所在的文件夹路径
        
    Returns:
        dict: 包含table_header和time_data的字典
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"在文件夹 {folder_path} 中未找到CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储所有表头（不包含TIME）
    all_headers = []
    # 存储每个文件的表头信息
    file_headers_info = []
    
    # 第一次遍历：收集所有表头，并确定所有时间点
    all_times = set()
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # 读取前几行确定分隔符
                sample = f.read(1024)
                f.seek(0)
                
                # 自动检测分隔符
                if '\t' in sample:
                    delimiter = '\t'
                elif ',' in sample:
                    delimiter = ','
                else:
                    delimiter = ','  # 默认逗号
                
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader)
                
                # 保存这个文件的表头信息
                file_headers = headers[1:]  # 跳过第一列（假设是TIME）
                file_headers_info.append({
                    'file': csv_file,
                    'headers': file_headers,
                    'delimiter': delimiter,
                    'start_index': len(all_headers)  # 这个文件的表头在总表头中的起始位置
                })
                
                # 添加到总表头
                all_headers.extend(file_headers)
                
                # 收集所有时间点
                for row in reader:
                    if row:  # 跳过空行
                        all_times.add(row[0])
                
                print(f"已读取文件 {os.path.basename(csv_file)}，表头数: {len(file_headers)}")
                
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")
            continue
    
    print(f"\n总表头数: {len(all_headers)}")
    print(f"总时间点数: {len(all_times)}")
    
    # 初始化数据结构
    time_data = {time: [None] * len(all_headers) for time in sorted(all_times)}
    
    # 第二次遍历：填充数据
    for file_info in file_headers_info:
        csv_file = file_info['file']
        start_index = file_info['start_index']
        num_headers = len(file_info['headers'])
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=file_info['delimiter'])
                next(reader)  # 跳过表头
                
                for row in reader:
                    if not row:
                        continue
                    
                    time_key = row[0]
                    if time_key not in time_data:
                        # 如果时间点不存在，创建它
                        time_data[time_key] = [None] * len(all_headers)
                    
                    # 填充这个时间点的数据
                    for i in range(num_headers):
                        value_index = start_index + i
                        if i + 1 < len(row):  # 确保有足够的数据
                            try:
                                # 转换值为浮点数
                                value = float(row[i + 1])
                            except (ValueError, TypeError):
                                value = None
                            time_data[time_key][value_index] = value
                
                print(f"已处理文件 {os.path.basename(csv_file)}，填充了 {num_headers} 列数据")
                
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            continue
    
    # 验证数据完整性
    valid_count = 0
    invalid_count = 0
    for time_key, values in time_data.items():
        if len(values) == len(all_headers):
            valid_count += 1
        else:
            invalid_count += 1
            print(f"警告: 时间点 {time_key} 的数据长度不正确: {len(values)} != {len(all_headers)}")
    
    print(f"\n数据验证结果:")
    print(f"有效时间点: {valid_count}")
    print(f"无效时间点: {invalid_count}")
    
    # 构建结果字典
    result = {
        "table_header": all_headers,
        "time_data": time_data
    }
    
    return result


def save_to_json_with_stats(data_dict: dict, output_path: str = "output.json"):
    """
    保存字典到JSON文件，并输出统计信息
    
    Args:
        data_dict: 要保存的字典
        output_path: 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON文件已保存到: {output_path}")
    
    # 输出统计信息
    table_header = data_dict.get("table_header", [])
    time_data = data_dict.get("time_data", {})
    
    print(f"\n统计信息:")
    print(f"表头总数: {len(table_header)}")
    print(f"时间点总数: {len(time_data)}")
    
    # 检查数据完整性
    if time_data:
        first_time = next(iter(time_data))
        first_values = time_data[first_time]
        print(f"第一个时间点 ({first_time}) 的数据长度: {len(first_values)}")
        
        # 统计None值比例
        total_values = len(table_header) * len(time_data)
        non_none_count = 0
        for values in time_data.values():
            non_none_count += sum(1 for v in values if v is not None)
        
        if total_values > 0:
            fill_rate = (non_none_count / total_values) * 100
            print(f"数据填充率: {fill_rate:.2f}%")
        
        # 查看前几个时间点的数据
        print(f"\n前3个时间点的数据摘要:")
        for i, (time_key, values) in enumerate(time_data.items()):
            if i >= 3:
                break
            none_count = sum(1 for v in values if v is None)
            print(f"  {time_key}: 数据长度={len(values)}, None值={none_count}")


def main():
    """
    主函数
    """
    # 配置参数
    folder_path = r"C:\Users\CXMO0\Documents\prediction_1hour"  # 请修改为实际的文件夹路径
    output_json_path = r"C:\Users\CXMO0\Documents\prediction_1hour\output\data.json"  # 输出JSON文件路径
    
    try:
        print("开始处理CSV文件...")
        
        # 处理CSV文件
        result_dict = process_csv_files_correctly(folder_path)
        
        # 保存为JSON文件
        save_to_json_with_stats(result_dict, output_json_path)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 提供一个更简单的版本，适合处理大量文件
def simple_process(folder_path: str):
    """
    简化的处理函数
    """
    import pandas as pd
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在 {folder_path} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 使用pandas读取和处理
    all_data = []
    all_headers = []
    
    for csv_file in csv_files:
        try:
            # 尝试自动检测分隔符
            df = pd.read_csv(csv_file, sep=None, engine='python', encoding='utf-8')
            
            # 获取表头（排除第一列，假设是时间列）
            headers = list(df.columns)[1:]
            all_headers.extend(headers)
            
            # 获取数据
            time_col = df.columns[0]
            data = df.iloc[:, 1:].values.tolist()  # 所有行，排除第一列
            
            all_data.append({
                'file': csv_file,
                'time': df[time_col].tolist(),
                'data': data,
                'headers': headers,
                'start_index': len(all_headers) - len(headers)
            })
            
            print(f"读取 {os.path.basename(csv_file)}: {df.shape[0]} 行, {len(headers)} 列")
            
        except Exception as e:
            print(f"读取 {csv_file} 失败: {e}")
    
    print(f"\n总表头数: {len(all_headers)}")
    
    # 确定所有时间点（假设所有文件的时间点相同）
    if not all_data:
        return None
    
    # 使用第一个文件的时间点
    all_times = all_data[0]['time']
    
    # 初始化数据结构
    time_data = {str(time): [None] * len(all_headers) for time in all_times}
    
    # 填充数据
    for file_info in all_data:
        times = file_info['time']
        data = file_info['data']
        start_index = file_info['start_index']
        num_cols = len(file_info['headers'])
        
        for i, time in enumerate(times):
            time_key = str(time)
            for j in range(num_cols):
                if j < len(data[i]):
                    time_data[time_key][start_index + j] = float(data[i][j]) if pd.notna(data[i][j]) else None
    
    # 构建结果
    result = {
        "table_header": all_headers,
        "time_data": time_data
    }
    
    return result


if __name__ == "__main__":
    # 使用方法1：使用正确的处理函数
    main()
    
    # 或者使用方法2：使用简化的pandas版本
    # folder_path = "你的csv文件夹路径"
    # result = simple_process(folder_path)
    # if result:
    #     with open("output_pandas.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, indent=2, ensure_ascii=False)
    #     print("已保存到 output_pandas.json")