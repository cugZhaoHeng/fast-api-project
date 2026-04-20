import os
import re
import pandas as pd

# ========== 配置区域 ==========
EXCEL_PATH = r"F:\贝乐虎儿歌100首\贝乐虎儿歌列表.xlsx"  # Excel文件路径
VIDEO_DIR = r"F:\贝乐虎儿歌100首"  # 视频文件所在目录（"."表示当前目录）
SHEET_NAME = 0  # 工作表名或索引（0表示第一个工作表）
COL_ID = 0  # 序号所在的列索引（0表示第一列）
COL_NAME = 1  # 歌名所在的列索引（1表示第二列）
VIDEO_EXT = ".mp4"  # 视频文件扩展名（小写）


# =============================

def extract_number(filename):
    """从文件名（不含扩展名）中提取开头的数字，返回int或None"""
    base = os.path.splitext(filename)[0]
    match = re.match(r'^(\d+)', base)
    return int(match.group(1)) if match else None


def main():
    # 1. 读取Excel建立序号→歌名的映射
    if not os.path.exists(EXCEL_PATH):
        print(f"错误：Excel文件不存在 - {EXCEL_PATH}")
        return

    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None, dtype=str)
        # 获取指定两列，去除空行
        id_col = df.iloc[:, COL_ID].astype(str).str.strip()
        name_col = df.iloc[:, COL_NAME].astype(str).str.strip()

        name_map = {}
        for idx, name in zip(id_col, name_col):
            if idx and name:  # 忽略空值
                try:
                    num = int(idx)  # 序号转为整数
                    name_map[num] = name
                except ValueError:
                    print(f"警告：跳过非数字序号 '{idx}'")

        if not name_map:
            print("错误：Excel中没有读取到有效的序号-歌名对应关系")
            return
        print(f"成功读取 {len(name_map)} 条映射记录")

    except Exception as e:
        print(f"读取Excel失败: {e}")
        return

    # 2. 遍历视频目录，处理所有匹配扩展名的文件
    if not os.path.isdir(VIDEO_DIR):
        print(f"错误：视频目录不存在 - {VIDEO_DIR}")
        return

    files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(VIDEO_EXT)]
    if not files:
        print(f"在目录 {VIDEO_DIR} 中没有找到 {VIDEO_EXT} 文件")
        return

    renamed_count = 0
    skipped_count = 0

    for old_name in files:
        old_path = os.path.join(VIDEO_DIR, old_name)
        num = extract_number(old_name)

        if num is None:
            print(f"跳过：文件名不含数字序号 - {old_name}")
            skipped_count += 1
            continue

        if num not in name_map:
            print(f"跳过：序号 {num} 不在Excel映射中 - {old_name}")
            skipped_count += 1
            continue

        # 构造新文件名
        new_name = f"{num:03d} {name_map[num]}{VIDEO_EXT}"
        new_path = os.path.join(VIDEO_DIR, new_name)

        # 避免覆盖
        if os.path.exists(new_path):
            print(f"警告：目标文件已存在，跳过 - {new_name}")
            skipped_count += 1
            continue

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"重命名: {old_name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"重命名失败: {old_name} -> {new_name}, 错误: {e}")
            skipped_count += 1

    print(f"\n完成！成功重命名 {renamed_count} 个文件，跳过 {skipped_count} 个文件。")


if __name__ == "__main__":
    main()