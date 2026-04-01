import pandas as pd
import os
import re
import numpy as np


def clean_column_name(col):
    """
    清理列名：去除引号、替换空格为下划线、处理换行符
    """
    if isinstance(col, tuple):  # 处理多级表头
        col = '_'.join([str(c) for c in col if str(c) != 'nan'])

    # 转换为字符串
    col = str(col)

    # 去除引号
    col = col.replace('"', '').replace("'", '')

    # 替换换行符和多个空格为下划线
    col = re.sub(r'\s+', '_', col)

    # 去除首尾的特殊字符
    col = col.strip('_')

    return col


def standardize_columns(df, reference_columns, fill_value=-9999):
    """
    根据参考列名标准化DataFrame的列
    缺失的列用指定值填充（默认-9999）
    """
    # 获取当前DataFrame的列名
    current_columns = list(df.columns)

    # 找出缺失的列（参考列中有，但当前DataFrame中没有的）
    missing_columns = [col for col in reference_columns if col not in current_columns]

    # 找出多余的列（当前DataFrame中有，但参考列中没有的）- 这些将被忽略
    extra_columns = [col for col in current_columns if col not in reference_columns]

    if missing_columns:
        print(f"    添加缺失列: {missing_columns}，用 {fill_value} 填充")
        for col in missing_columns:
            df[col] = fill_value

    if extra_columns:
        print(f"    忽略多余列: {extra_columns}")

    # 重新排列列的顺序，使其与参考列一致
    df = df[reference_columns]

    return df


def select_excel_file():
    """
    让用户选择要处理的Excel文件
    """
    # 获取当前目录下所有Excel文件
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        print("当前目录下没有找到Excel文件！")
        return None

    print("\n找到以下Excel文件：")
    for i, file in enumerate(excel_files, 1):
        print(f"  {i}. {file}")

    while True:
        try:
            choice = input("\n请选择要处理的文件编号（输入数字）：")
            idx = int(choice) - 1
            if 0 <= idx < len(excel_files):
                return excel_files[idx]
            else:
                print(f"请输入1-{len(excel_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")


def ask_yes_no(question):
    """
    询问用户是/否问题
    """
    while True:
        answer = input(f"{question} (y/n): ").lower().strip()
        if answer in ['y', 'yes', '是']:
            return True
        elif answer in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y 或 n")


def format_value_for_txt(val, decimal_places=3):
    """
    格式化数值用于TXT输出
    """
    if isinstance(val, (int, float)):
        if val == -9999:
            return "-9999"
        elif isinstance(val, float):
            return f"{val:.{decimal_places}f}"
        else:
            return str(val)
    else:
        return str(val)


def excel_to_txt_interactive():
    """
    交互式Excel转TXT（空格分隔），所有sheet保持相同的列顺序，缺失值用-9999填充
    """
    print("=" * 50)
    print("Excel转TXT工具（空格分隔）")
    print("=" * 50)

    # 1. 选择Excel文件
    excel_file = select_excel_file()
    if not excel_file:
        return

    # 2. 询问是否添加sheet名作为well_name
    add_well_name = ask_yes_no("\n是否将sheet名作为well_name添加到第一列？")

    # 3. 询问小数位数
    while True:
        try:
            decimal_places = input("\n请输入要保留的小数位数（直接回车默认3位）：").strip()
            if decimal_places == '':
                decimal_places = 3
                break
            decimal_places = int(decimal_places)
            if decimal_places >= 0:
                break
            else:
                print("请输入非负整数")
        except ValueError:
            print("请输入有效的数字")

    print("\n开始处理...")
    print("-" * 50)

    try:
        # 获取Excel文件名（不含路径和扩展名）
        base_name = os.path.splitext(os.path.basename(excel_file))[0]

        # 创建以Excel文件名命名的文件夹
        folder_name = base_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"创建文件夹: {folder_name}")

        # 读取Excel文件
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names

        print(f"\n处理文件: {excel_file}")
        print(f"发现 {len(sheet_names)} 个sheet: {sheet_names}")

        # 存储所有sheet的数据和参考列名
        all_sheets_data = {}
        reference_columns = None
        first_sheet_with_data = None

        # 第一次遍历：读取所有sheet并确定参考列名
        print("\n第一次遍历：分析所有sheet的列结构...")
        for sheet in sheet_names:
            try:
                # 读取sheet数据
                df = pd.read_excel(excel_file, sheet_name=sheet, header=0)

                if df.empty:
                    print(f"  {sheet}: 空sheet，跳过")
                    continue

                print(f"  {sheet}: 原始数据形状 {df.shape}, 列名: {list(df.columns)}")

                # 如果需要添加well_name列，先添加再清理列名
                if add_well_name:
                    df.insert(0, 'well_name', sheet)

                # 清理列名
                df.columns = [clean_column_name(col) for col in df.columns]

                # 存储清理后的数据
                all_sheets_data[sheet] = df

                # 如果是第一个非空sheet，设置为参考列名
                if reference_columns is None:
                    reference_columns = list(df.columns)
                    first_sheet_with_data = sheet
                    print(f"  设置参考列名（来自 {sheet}）: {reference_columns}")

            except Exception as e:
                print(f"  读取sheet '{sheet}' 时出错: {e}")

        if reference_columns is None:
            print("错误：没有找到有效的sheet数据！")
            return

        print(f"\n参考列名（以 {first_sheet_with_data} 为准）:")
        for i, col in enumerate(reference_columns, 1):
            print(f"  {i}. {col}")

        print("\n第二次遍历：标准化并导出所有sheet...")

        # 第二次遍历：标准化并导出
        converted_count = 0
        for sheet in sheet_names:
            try:
                if sheet not in all_sheets_data:
                    continue

                df = all_sheets_data[sheet].copy()  # 使用副本避免修改原数据

                if df.empty:
                    continue

                print(f"\n处理sheet: {sheet}")
                print(f"  标准化前列名: {list(df.columns)}")

                # 标准化列，缺失值用-9999填充
                df = standardize_columns(df, reference_columns, fill_value=-9999)

                print(f"  标准化后列名: {list(df.columns)}")

                # 清理sheet名中的非法字符（用于文件名）
                safe_sheet_name = re.sub(r'[<>:"/\\|?*]', '_', sheet)
                txt_filename = os.path.join(folder_name, f"{safe_sheet_name}.txt")

                # 手动写入TXT文件（空格分隔）
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    # 写入表头
                    header_line = ' '.join(df.columns)
                    f.write(header_line + '\n')

                    # 写入数据行
                    for _, row in df.iterrows():
                        # 格式化每个值
                        formatted_values = []
                        for val in row:
                            if isinstance(val, (int, float)):
                                if val == -9999:
                                    formatted_values.append("-9999")
                                elif isinstance(val, float):
                                    formatted_values.append(f"{val:.{decimal_places}f}")
                                else:
                                    formatted_values.append(str(val))
                            else:
                                formatted_values.append(str(val))

                        # 写入一行数据
                        data_line = ' '.join(formatted_values)
                        f.write(data_line + '\n')

                print(f"  ✓ 已保存: {safe_sheet_name}.txt ({len(df)} 行, {len(df.columns)} 列)")

                # 显示前几行数据预览
                print(f"  数据预览（前2行）:")
                with open(txt_filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[:3]):  # 显示表头+前2行数据
                        print(f"    {line.strip()}")

                converted_count += 1

            except Exception as e:
                print(f"  ✗ 处理sheet '{sheet}' 时出错: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n完成！已转换 {converted_count}/{len(sheet_names)} 个sheet")
        print(f"文件保存在文件夹: {folder_name}")

    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)


# 快速导出函数
def quick_export_to_txt(excel_file, add_well_name=False, decimal_places=3):
    """
    快速导出指定Excel文件为TXT（空格分隔）
    """
    try:
        base_name = os.path.splitext(os.path.basename(excel_file))[0]
        folder_name = base_name

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names

        print(f"\n处理文件: {excel_file}")

        # 存储所有sheet的数据
        all_sheets_data = {}
        reference_columns = None

        # 第一次遍历：读取并清理所有sheet
        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet, header=0)

            if df.empty:
                continue

            if add_well_name:
                df.insert(0, 'well_name', sheet)

            df.columns = [clean_column_name(col) for col in df.columns]
            all_sheets_data[sheet] = df

            if reference_columns is None:
                reference_columns = list(df.columns)
                print(f"  参考列名（来自 {sheet}）: {reference_columns}")

        if reference_columns is None:
            print("  没有有效数据")
            return False

        # 第二次遍历：标准化并导出
        for sheet, df in all_sheets_data.items():
            # 标准化列
            df = standardize_columns(df, reference_columns, fill_value=-9999)

            # 保存为TXT文件
            safe_sheet_name = re.sub(r'[<>:"/\\|?*]', '_', sheet)
            txt_filename = os.path.join(folder_name, f"{safe_sheet_name}.txt")

            with open(txt_filename, 'w', encoding='utf-8') as f:
                # 写入表头
                header_line = ' '.join(df.columns)
                f.write(header_line + '\n')

                # 写入数据行
                for _, row in df.iterrows():
                    formatted_values = []
                    for val in row:
                        if isinstance(val, (int, float)):
                            if val == -9999:
                                formatted_values.append("-9999")
                            elif isinstance(val, float):
                                formatted_values.append(f"{val:.{decimal_places}f}")
                            else:
                                formatted_values.append(str(val))
                        else:
                            formatted_values.append(str(val))

                    data_line = ' '.join(formatted_values)
                    f.write(data_line + '\n')

            print(f"  ✓ {safe_sheet_name}.txt ({len(df)} 行)")

        print(f"完成！文件保存在: {folder_name}")
        return True

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 交互式模式
    excel_to_txt_interactive()

    # 快速导出示例（注释掉的）
    # quick_export_to_txt("你的文件.xlsx", add_well_name=True, decimal_places=3)