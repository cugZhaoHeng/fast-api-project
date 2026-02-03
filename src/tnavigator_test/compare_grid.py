import os
import sys
from typing import List, Tuple

def parse_zcorn_data(file_path: str) -> List[float]:
    """
    解析ZCONN/PORO格式数据文件，展开所有数值
    
    参数:
    file_path: 数据文件路径
    
    返回:
    List[float]: 展开后的数值列表
    """
    values = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除首尾空白字符
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                    
                # 跳过注释行和标题行
                # 检查是否以数字、负号或小数点开头
                stripped_line = line.lstrip()
                if not stripped_line:
                    continue
                    
                first_char = stripped_line[0]
                if not (first_char.isdigit() or first_char == '-' or first_char == '.'):
                    # 不是数据行，跳过
                    continue
                
                # 按空格分割数据项
                items = stripped_line.split()
                
                for item in items:
                    # 跳过非数字开头的项
                    if not item:
                        continue
                    
                    first_char_item = item[0]
                    if not (first_char_item.isdigit() or first_char_item == '-' or first_char_item == '.'):
                        continue
                        
                    # 检查是否包含乘号
                    if '*' in item:
                        # 分割乘号左右部分
                        if item.startswith('*'):
                            # 处理以*开头的情况
                            count = 1
                            value_str = item[1:]
                        else:
                            parts = item.split('*')
                            if len(parts) == 2:
                                try:
                                    count = int(parts[0])
                                    value_str = parts[1]
                                except ValueError:
                                    # 如果不能解析为整数，默认为1
                                    count = 1
                                    value_str = parts[1]
                            else:
                                # 格式不正确，跳过
                                continue
                        
                        # 展开重复值
                        try:
                            value = float(value_str)
                            values.extend([value] * count)
                        except ValueError:
                            # 如果不能转换为浮点数，跳过
                            continue
                    else:
                        # 没有乘号，直接添加
                        try:
                            value = float(item)
                            values.append(value)
                        except ValueError:
                            # 如果不能转换为浮点数，跳过
                            continue
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        raise
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise
    
    return values


def compare_data_files(file1_path: str, file2_path: str, output_path: str, 
                      threshold: float = 0.001, print_interval: int = 10000) -> Tuple[int, float, float, float]:
    """
    比较两个数值数据文件的差异
    
    参数:
    file1_path: 第一个文件路径
    file2_path: 第二个文件路径
    output_path: 输出结果文件路径
    threshold: 差异阈值，超过此阈值才记录为显著差异
    print_interval: 打印进度的间隔
    
    返回:
    Tuple[int, float, float, float]: 
        - 总数据个数
        - 平均绝对差异
        - 最大绝对差异
        - 差异超过阈值的百分比
    """
    print(f"正在解析第一个文件: {file1_path}")
    data1 = parse_zcorn_data(file1_path)
    print(f"第一个文件解析完成，共 {len(data1)} 个数据")
    
    print(f"正在解析第二个文件: {file2_path}")
    data2 = parse_zcorn_data(file2_path)
    print(f"第二个文件解析完成，共 {len(data2)} 个数据")
    
    # 检查数据个数是否一致
    if len(data1) != len(data2):
        print(f"警告: 两个文件的数据个数不一致! 文件1: {len(data1)}, 文件2: {len(data2)}")
        print("将比较较短文件长度的数据")
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
    else:
        min_length = len(data1)
    
    print(f"开始比较 {min_length} 个数据...")
    
    # 初始化统计信息
    total_diff = 0.0
    max_abs_diff = 0.0
    max_diff_index = 0
    above_threshold_count = 0
    
    # 用于存储差异详情
    diff_details = []
    
    # 比较每个数据
    for i in range(min_length):
        # 计算绝对差异
        abs_diff = abs(data1[i] - data2[i])
        
        # 更新统计信息
        total_diff += abs_diff
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_diff_index = i
        
        # 检查是否超过阈值
        if abs_diff > threshold:
            above_threshold_count += 1
            diff_details.append((
                i + 1,  # 索引从1开始
                data1[i],
                data2[i],
                abs_diff,
                data1[i] - data2[i]  # 有符号差异
            ))
        
        # 打印进度
        if (i + 1) % print_interval == 0:
            print(f"已处理 {i + 1}/{min_length} 个数据...")
    
    # 计算统计信息
    avg_abs_diff = total_diff / min_length if min_length > 0 else 0.0
    above_threshold_percentage = (above_threshold_count / min_length) * 100 if min_length > 0 else 0.0
    
    # 输出结果到文件
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            # 写入文件信息
            output_file.write(f"文件比较结果\n")
            output_file.write(f"=" * 80 + "\n\n")
            output_file.write(f"文件1: {file1_path}\n")
            output_file.write(f"文件2: {file2_path}\n")
            output_file.write(f"总数据个数: {min_length}\n")
            output_file.write(f"\n")
            
            # 写入统计摘要
            output_file.write("统计摘要:\n")
            output_file.write(f"- 平均绝对差异: {avg_abs_diff:.6e}\n")
            output_file.write(f"- 最大绝对差异: {max_abs_diff:.6e} (位置: {max_diff_index + 1})\n")
            output_file.write(f"- 差异超过阈值({threshold})的个数: {above_threshold_count}\n")
            output_file.write(f"- 差异超过阈值({threshold})的百分比: {above_threshold_percentage:.2f}%\n")
            
            # 写入最大值详情
            if max_abs_diff > 0:
                output_file.write(f"\n最大差异详情:\n")
                output_file.write(f"- 位置: {max_diff_index + 1}\n")
                output_file.write(f"- 文件1值: {data1[max_diff_index]}\n")
                output_file.write(f"- 文件2值: {data2[max_diff_index]}\n")
                output_file.write(f"- 绝对差异: {max_abs_diff}\n")
                output_file.write(f"- 相对差异: {abs((data1[max_diff_index] - data2[max_diff_index]) / data1[max_diff_index]) * 100 if data1[max_diff_index] != 0 else 'INF':.2f}%\n")
            
            # 写入差异超过阈值的详情
            if diff_details:
                output_file.write(f"\n差异超过阈值({threshold})的详细信息 (共{len(diff_details)}个):\n")
                output_file.write("-" * 100 + "\n")
                output_file.write(f"{'位置':<10} {'文件1值':<20} {'文件2值':<20} {'绝对差异':<20} {'有符号差异':<20}\n")
                output_file.write("-" * 100 + "\n")
                
                for detail in diff_details:
                    position, val1, val2, abs_diff, signed_diff = detail
                    output_file.write(f"{position:<10} {val1:<20.6e} {val2:<20.6e} {abs_diff:<20.6e} {signed_diff:<20.6e}\n")
            
            # 如果数据完全一致
            if max_abs_diff == 0:
                output_file.write(f"\n两个文件的数据完全一致!\n")
    
    except Exception as e:
        print(f"写入输出文件时出错: {e}")
        raise
    
    # 打印摘要到控制台
    print("\n" + "=" * 60)
    print("比较完成!")
    print("=" * 60)
    print(f"总数据个数: {min_length}")
    print(f"平均绝对差异: {avg_abs_diff:.6e}")
    print(f"最大绝对差异: {max_abs_diff:.6e} (位置: {max_diff_index + 1})")
    print(f"差异超过阈值({threshold})的个数: {above_threshold_count}")
    print(f"差异超过阈值({threshold})的百分比: {above_threshold_percentage:.2f}%")
    print(f"详细结果已保存到: {output_path}")
    print("=" * 60)
    
    return min_length, avg_abs_diff, max_abs_diff, above_threshold_percentage

# 使用示例
if __name__ == "__main__":
    
    try:
        # 创建测试文件
        property_name = "poro"
        file1_path = rf"D:\git\fast-api-project\src\tnavigator_test\{property_name}-grdecl.inc"
        file2_path = rf"D:\git\fast-api-project\src\tnavigator_test\{property_name}-tna.inc"
        output_path = rf"D:\git\fast-api-project\src\tnavigator_test\{property_name}-comparison-result2.txt"
        
        # 比较文件
        result = compare_data_files(file1_path, file2_path, output_path, threshold=0.0001)
        
        print(f"\n比较结果:")
        print(f"- 总数据个数: {result[0]}")
        print(f"- 平均绝对差异: {result[1]:.6e}")
        print(f"- 最大绝对差异: {result[2]:.6e}")
        print(f"- 差异超过阈值的百分比: {result[3]:.2f}%")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
    
    print("\n" + "=" * 60)
    print("示例2: 使用实际文件")
    print("=" * 60)