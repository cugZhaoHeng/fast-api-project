from datetime import datetime
import os
from pathlib import Path
from typing import List

current_dir = Path(__file__).parent

def convert_dates_from_file(input_file_path, output_file_path):
    """
    从 input_file_path 读取每行日期（格式如 2022/1/23），
    转换为 dd.MM.yyyy 格式，并写入 output_file_path。
    
    参数:
        input_file_path (str): 输入文件路径（包含原始日期）
        output_file_path (str): 输出文件路径（保存转换后的日期）
    """
    if not os.path.exists(input_file_path):
        print(f"❌ 错误：输入文件 '{input_file_path}' 不存在。")
        return

    converted_lines = []
    error_lines = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, start=1):
            original = line.strip()
            if not original:
                continue  # 跳过空行

            try:
                dt = datetime.strptime(original, "%Y/%m/%d")
                formatted = dt.strftime("%d.%m.%Y")
                converted_lines.append(formatted)
            except ValueError:
                error_lines.append((line_num, original))

    # 写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in converted_lines:
            outfile.write(line + '\n')

    # 打印处理摘要
    print(f"✅ 成功转换 {len(converted_lines)} 行日期，已保存到 '{output_file_path}'")
    if error_lines:
        print("⚠️ 以下行格式有误，已被跳过：")
        for line_num, content in error_lines:
            print(f"  第 {line_num} 行: '{content}'")


def extract_lines_to_file(input_file_path, output_file_path, start_line, end_line):
    """
    从输入文件中提取指定行范围的内容，写入到输出文件中
    
    参数:
    input_file_path (str): 输入文件路径
    output_file_path (str): 输出文件路径
    start_line (int): 起始行号（从1开始）
    end_line (int): 结束行号（包含在输出中）
    
    返回:
    bool: 操作是否成功
    """
    try:
        # 验证行号参数
        if start_line < 1 or end_line < 1:
            print("错误：行号必须大于等于1")
            return False
        
        if start_line > end_line:
            print("错误：起始行号不能大于结束行号")
            return False
        
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
        
        # 检查行号是否有效
        if end_line > len(lines):
            print(f"警告：结束行号{end_line}超过文件总行数{len(lines)}，将提取到文件末尾")
            end_line = len(lines)
        
        # 提取指定范围的行（注意列表索引从0开始）
        content_to_extract = lines[start_line-1:end_line]
        
        # 写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(content_to_extract)
        
        # 显示提取信息
        extracted_lines = len(content_to_extract)
        print(f"成功提取 {extracted_lines} 行内容")
        print(f"从行 {start_line} 到行 {end_line}")
        print(f"输出文件: {output_file_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file_path}'")
        return False
    except PermissionError:
        print(f"错误：没有权限访问文件 '{input_file_path}'")
        return False
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return False

def process_zcorn_data(input_file_path: str, output_file_path: str):
    """
    处理ZCONN/PORO格式数据文件，将所有数值除以100
    
    参数:
    input_file_path: 输入文件路径
    output_file_path: 输出文件路径
    """
    import re
    
    try:
        with open(input_file_path, 'r') as input_file:
            with open(output_file_path, 'w') as output_file:
                for line in input_file:
                    # 如果是注释行或空行，直接写入
                    if not line.strip() or line.strip().startswith('--'):
                        output_file.write(line)
                        continue
                    
                    # 处理行中的每个数据项
                    # 方法：分割行，逐个处理每个部分
                    parts = line.strip().split()
                    processed_parts = []
                    
                    for part in parts:
                        # 检查是否包含乘号
                        if '*' in part:
                            star_parts = part.split('*')
                            if len(star_parts) == 2:
                                count_part = star_parts[0]
                                value_part = star_parts[1]
                                
                                # 处理乘号左边为空的情况（如 "*0"）
                                if not count_part:
                                    count_part = ""
                                
                                try:
                                    # 将数值除以100
                                    original_value = float(value_part)
                                    new_value = original_value / 100
                                    
                                    # 格式化
                                    new_value_str = f"{new_value:.6f}".rstrip('0').rstrip('.')
                                    if new_value_str == "-0":
                                        new_value_str = "0"
                                    
                                    processed_parts.append(f"{count_part}*{new_value_str}")
                                except ValueError:
                                    processed_parts.append(part)
                            else:
                                # 格式不正确，保持原样
                                processed_parts.append(part)
                        else:
                            # 没有乘号，尝试将整个部分作为数值处理
                            try:
                                original_value = float(part)
                                new_value = original_value / 100
                                
                                # 格式化
                                new_value_str = f"{new_value:.6f}".rstrip('0').rstrip('.')
                                if new_value_str == "-0":
                                    new_value_str = "0"
                                
                                processed_parts.append(new_value_str)
                            except ValueError:
                                # 不是数值，保持原样
                                processed_parts.append(part)
                    
                    # 重新组合行
                    # 保留原行的缩进
                    leading_spaces = len(line) - len(line.lstrip())
                    processed_line = ' ' * leading_spaces + ' '.join(processed_parts)
                    
                    # 如果原行有换行符，保持它
                    if line.endswith('\n'):
                        processed_line += '\n'
                    
                    output_file.write(processed_line)
        
        print(f"处理完成！结果已保存到: {output_file_path}")
        
    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 不存在")
        raise
    except Exception as e:
        print(f"处理文件时出错: {e}")
        raise

def replace_file_section(source_file: str, target_file: str, start_line: int, end_line: int, output_file: str = None):
    """
    用源文件的内容替换目标文件指定行范围的内容
    
    参数:
    source_file: 源文件路径，提供替换内容
    target_file: 目标文件路径，将被替换部分内容
    start_line: 起始行号（从1开始，包含）
    end_line: 结束行号（包含）
    output_file: 输出文件路径，如为None则覆盖原目标文件
    """
    try:
        # 验证参数
        if start_line < 1 or end_line < 1:
            raise ValueError("行号必须大于等于1")
        if start_line > end_line:
            raise ValueError("起始行号不能大于结束行号")
        
        # 读取源文件内容
        with open(source_file, 'r', encoding='utf-8') as f:
            source_content = f.read()
        
        # 读取目标文件内容
        with open(target_file, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        
        # 检查行号是否有效
        if end_line > len(target_lines):
            print(f"警告：结束行号{end_line}超过文件总行数{len(target_lines)}")
            end_line = len(target_lines)
        
        # 构建新文件内容
        new_lines = []
        
        # 1. 添加起始行之前的内容
        new_lines.extend(target_lines[:start_line-1])
        
        # 2. 添加源文件内容
        # 注意：源文件内容可能有多行，需要按行分割
        source_lines = source_content.splitlines(keepends=True)
        
        # 如果源文件内容没有以换行符结尾，且不是空内容，需要添加换行符
        if source_content and not source_content.endswith('\n'):
            # 检查是否需要添加换行符
            if end_line < len(target_lines):
                source_lines[-1] = source_lines[-1].rstrip('\n') + '\n'
        
        new_lines.extend(source_lines)
        
        # 3. 添加结束行之后的内容
        new_lines.extend(target_lines[end_line:])
        
        # 确定输出文件路径
        if output_file is None:
            output_file = target_file
            print(f"将覆盖原文件: {target_file}")
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"替换完成！")
        print(f"目标文件: {target_file}")
        print(f"源文件: {source_file}")
        print(f"替换行范围: {start_line}-{end_line}")
        print(f"输出文件: {output_file}")
        print(f"源文件行数: {len(source_lines)}")
        print(f"替换后总行数: {len(new_lines)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}")
        return False
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    input_path = current_dir / "data" / "LJZ.GRDECL"
    output_path = current_dir / "data" / "LJZ-SWAT.GRDECL"
    extract_lines_to_file(input_path, output_path, 171859, 179351)
    
    input_path = current_dir / "data" / "LJZ-SWAT.GRDECL"
    output_path = current_dir / "data" / "LJZ-SWAT-new.GRDECL"
    process_zcorn_data(input_path, output_path)
    
    replace_file_section(
        source_file=current_dir / "data" / "LJZ-SWAT-new.GRDECL",
        target_file=current_dir / "data" / "LJZ.GRDECL",
        start_line=171859,
        end_line=179351,
        output_file=current_dir / "data" / "LJZ-new.GRDECL"
    )