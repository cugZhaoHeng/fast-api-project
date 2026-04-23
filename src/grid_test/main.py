import re

def process_swat_file(input_path, output_path):
    with open(input_path, 'r') as f:
        content = f.read()

    # 匹配两种情况：
    # 1. N*V 形式
    # 2. 单个数字（浮点/整数）
    pattern = re.compile(r'(\d+)\*(\-?\d+\.?\d*)|(\-?\d+\.?\d*)')

    def replace_func(match):
        # 情况1：N*V
        if match.group(1) and match.group(2):
            n = match.group(1)
            v = float(match.group(2)) / 100
            return f"{n}*{v:.6f}"
        # 情况2：单个值
        elif match.group(3):
            v = float(match.group(3)) / 100
            return f"{v:.6f}"
        return match.group(0)

    # 只处理 SWAT 到 "/" 之间的内容
    def process_block(text):
        return pattern.sub(replace_func, text)

    # 找到 SWAT 块
    result = ""
    in_swat = False

    for line in content.splitlines():
        if line.strip().startswith("PORO"):
            in_swat = True
            result += line + "\n"
            continue

        if in_swat:
            if "/" in line:
                # 处理最后一行（包含 /）
                parts = line.split("/")
                processed = process_block(parts[0])
                result += processed + " /\n"
                in_swat = False
            else:
                result += process_block(line) + "\n"
        else:
            result += line + "\n"

    with open(output_path, 'w') as f:
        f.write(result)


# 使用示例
# process_swat_file("input.txt", "output.txt")
process_swat_file("input_poro.txt", "output_poro.txt")