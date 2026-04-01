import os
import re

def rename_baby_bus_songs(folder_path):
    """
    将指定文件夹中的MP4文件名从
    “宝宝巴士儿歌：第X话 歌名.mp4”
    改为
    “XXX 歌名.mp4”（XXX为001、002...按原数字顺序）
    """
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 正则匹配模式
    pattern = re.compile(r'宝宝巴士儿歌：第(\d+)话 (.+\.mp4)$')
    files = []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))          # 提取数字部分
            song = match.group(2)              # 提取歌名+扩展名
            full_path = os.path.join(folder_path, filename)
            files.append((num, song, full_path))
        else:
            print(f"跳过不匹配的文件: {filename}")

    if not files:
        print("没有找到匹配的MP4文件。")
        return

    # 按原始数字排序
    files.sort(key=lambda x: x[0])

    # 重命名
    for idx, (_, song, old_path) in enumerate(files, start=1):
        new_name = f"{idx:03d} {song}"
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(new_path):
            print(f"警告：目标文件 '{new_name}' 已存在，跳过重命名 {os.path.basename(old_path)}")
            continue

        try:
            os.rename(old_path, new_path)
            print(f"重命名: {os.path.basename(old_path)} -> {new_name}")
        except Exception as e:
            print(f"重命名失败: {old_path} -> {new_name}, 错误: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input("请输入宝宝巴士文件夹的路径: ").strip()
        if not folder:
            print("未提供路径，退出。")
            sys.exit(1)
    rename_baby_bus_songs(folder)