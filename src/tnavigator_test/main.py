def count_zcorn_data(file_path):
    """
    统计ZCORN格式数据文件中的数值个数
    
    参数:
    file_path: 数据文件路径
    
    返回:
    int: 数据的总个数
    """
    total_count = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除首尾空白字符
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                
                if line.startswith('--'):
                    continue
                    
                # 检查是否以数字开头（包括负号开头的情况）
                # 先去除行首的空格，然后检查第一个字符
                stripped_line = line.lstrip()
                if not stripped_line:
                    continue
                    
                first_char = stripped_line[0]
                # 检查第一个字符是否是数字或负号（用于负数）
                if not (first_char.isdigit() or first_char == '-' or first_char == '.'):
                    # 如果不是以数字、负号或小数点开头，跳过该行
                    continue
                
                # 按空格分割数据项
                items = line.split()
                
                for item in items:
                    # 跳过非数字开头的项（额外的安全措施）
                    if not (item[0].isdigit() or item[0] == '-' or item[0] == '.'):
                        continue
                        
                    # 检查是否包含乘号
                    if '*' in item:
                        # 分割乘号左右部分
                        if item.startswith('*'):
                            # 处理以*开头的情况（如*4866.070，表示1*4866.070）
                            count = 1
                            value = item[1:]
                        else:
                            parts = item.split('*')
                            if len(parts) == 2:
                                # 确保左侧是数字
                                if parts[0].isdigit():
                                    count = int(parts[0])
                                else:
                                    # 如果左侧不是纯数字，按1处理
                                    count = 1
                                value = parts[1]
                            else:
                                # 格式不正确，跳过
                                continue
                    else:
                        # 没有乘号，计数为1
                        count = 1
                        value = item
                    # 累加计数
                    total_count += count
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return -1
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return -1
    
    return total_count


# 使用示例
if __name__ == "__main__":
    # 运行详细测试
    count = count_zcorn_data(r'D:\git\fast-api-project\src\tnavigator_test\a.inc')
    print(f"ZCORN数据总个数: {count}")