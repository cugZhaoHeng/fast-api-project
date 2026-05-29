from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. 修改这里：服务器上的 CSV 文件夹
# ============================================================
CSV_DIR = Path(r"/home/tet/zhaoheng/fast-api-project/data/2023_log_csv")

# 是否递归处理子文件夹
RECURSIVE = False

# 是否覆盖原文件
# True  = 原地覆盖，并自动生成 .bak 备份
# False = 不覆盖原文件，输出到同级新文件夹 xxx_cleaned
OVERWRITE = True


# ============================================================
# 2. 异常值规则
# ============================================================

# LAS / CSV 中常见的无效占位值
COMMON_INVALID_VALUES = [
    -999.25,
    -999.0,
    -9999.0,
    -99999.0,
    9999.0,
    99999.0,
    10000.0,
    100000.0,
]

# 各测井曲线合理物理范围
# 范围外统一处理为 NaN
VALID_RANGES = {
    # 自然伽马，API
    "GR": {
        "min": 0.0,
        "max": 300.0,
        "include_min": True,
        "include_max": True,
    },

    # 体积密度，g/cm3
    "RHOB": {
        "min": 1.0,
        "max": 4.0,
        "include_min": True,
        "include_max": True,
    },

    # 深侧向电阻率，ohm·m
    # 注意：RILD = 10000 / 100000 这类值通常是异常占位值
    "RILD": {
        "min": 0.0,
        "max": 10000.0,
        "include_min": False,
        "include_max": False,
    },

    # 中子孔隙度，通常应为小数形式
    # 如果原数据是百分数，比如 25，会自动除以 100
    "CNPOR": {
        "min": -0.2,
        "max": 1.0,
        "include_min": True,
        "include_max": True,
    },
}


def find_col(df: pd.DataFrame, target: str):
    """
    忽略大小写匹配列名。
    例如 RILD / rild / Rild 都可以匹配。
    """
    col_map = {c.upper(): c for c in df.columns}
    return col_map.get(target.upper())


def replace_common_invalid_values(s: pd.Series) -> pd.Series:
    """
    先把常见占位异常值替换成 NaN。
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace(COMMON_INVALID_VALUES, np.nan)
    return s


def normalize_cnp_or_if_percent(s: pd.Series) -> pd.Series:
    """
    CNPOR 有些文件可能是百分数，如 25 表示 0.25。
    判断逻辑：
    - 先去掉 NaN；
    - 如果中位数 > 1.5，基本可以认为是百分数；
    - 则整体除以 100。
    """
    valid = s.dropna()
    if len(valid) > 0 and valid.median() > 1.5:
        s = s / 100.0
    return s


def apply_physical_range(s: pd.Series, log_name: str):
    """
    按物理范围过滤。
    返回：
        cleaned_series, invalid_count_by_range
    """
    rule = VALID_RANGES[log_name]
    before_na = s.isna().sum()

    min_v = rule["min"]
    max_v = rule["max"]

    if rule["include_min"]:
        low_mask = s < min_v
    else:
        low_mask = s <= min_v

    if rule["include_max"]:
        high_mask = s > max_v
    else:
        high_mask = s >= max_v

    invalid_mask = low_mask | high_mask
    invalid_count = int(invalid_mask.sum())

    s = s.mask(invalid_mask, np.nan)

    after_na = s.isna().sum()
    added_na = int(after_na - before_na)

    return s, invalid_count, added_na


def clean_one_dataframe(df: pd.DataFrame, csv_name: str):
    """
    清洗一个 CSV 的 DataFrame。
    """
    df = df.copy()

    report = {
        "file": csv_name,
        "changed": False,
        "detail": {},
    }

    # Depth 只做数值转换，不做范围过滤
    depth_col = find_col(df, "Depth")
    if depth_col is not None:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")

    for log_name in VALID_RANGES.keys():
        col = find_col(df, log_name)

        if col is None:
            report["detail"][log_name] = {
                "status": "missing column",
                "common_invalid": 0,
                "range_invalid": 0,
                "nan_total": None,
            }
            continue

        original = df[col].copy()

        # 1. 数值化 + 常见占位值转 NaN
        s0 = pd.to_numeric(df[col], errors="coerce")
        na_before = int(s0.isna().sum())

        s = replace_common_invalid_values(s0)
        common_invalid_count = int(s.isna().sum() - na_before)

        # 2. CNPOR 百分数转小数
        if log_name == "CNPOR":
            s = normalize_cnp_or_if_percent(s)

        # 3. 物理范围过滤
        s, range_invalid_count, _ = apply_physical_range(s, log_name)

        df[col] = s

        changed = not df[col].equals(original)
        if changed:
            report["changed"] = True

        report["detail"][log_name] = {
            "status": "ok",
            "common_invalid": common_invalid_count,
            "range_invalid": range_invalid_count,
            "nan_total": int(df[col].isna().sum()),
            "valid_total": int(df[col].notna().sum()),
        }

    return df, report


def process_one_csv(csv_path: Path, output_dir: Path | None = None):
    try:
        df = pd.read_csv(csv_path)
        cleaned_df, report = clean_one_dataframe(df, csv_path.name)

        if not report["changed"]:
            print(f"[OK] {csv_path.name}: 无需修改")
            return report

        if OVERWRITE:
            bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")

            # 第一次处理时备份原文件；如果备份已存在，不重复覆盖备份
            if not bak_path.exists():
                csv_path.replace(bak_path)
            else:
                print(f"[WARN] {csv_path.name}: 备份已存在，跳过重新备份")

            cleaned_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            print(f"[FIXED] {csv_path.name}: 已清洗并覆盖原文件，备份文件: {bak_path.name}")
        else:
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / csv_path.name
            cleaned_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"[FIXED] {csv_path.name}: 已清洗，输出到: {save_path}")

        for log_name, info in report["detail"].items():
            if info["status"] == "ok":
                print(
                    f"    {log_name:<5} | "
                    f"占位异常: {info['common_invalid']:<6} | "
                    f"范围异常: {info['range_invalid']:<6} | "
                    f"NaN总数: {info['nan_total']:<6} | "
                    f"有效值: {info['valid_total']}"
                )
            else:
                print(f"    {log_name:<5} | 缺少该列")

        return report

    except Exception as e:
        print(f"[FAILED] {csv_path.name}: {e}")
        return {
            "file": csv_path.name,
            "changed": False,
            "failed": True,
            "error": str(e),
            "detail": {},
        }


def main():
    if RECURSIVE:
        csv_files = sorted(CSV_DIR.rglob("*.csv")) + sorted(CSV_DIR.rglob("*.CSV"))
    else:
        csv_files = sorted(CSV_DIR.glob("*.csv")) + sorted(CSV_DIR.glob("*.CSV"))

    print("=" * 90)
    print(f"CSV_DIR = {CSV_DIR}")
    print(f"发现 CSV 文件数量: {len(csv_files)}")
    print(f"OVERWRITE = {OVERWRITE}")
    print("清洗规则：")
    print("  GR    : 0 <= GR <= 300")
    print("  RHOB  : 1.0 <= RHOB <= 4.0")
    print("  RILD  : 0 < RILD < 10000")
    print("  CNPOR : -0.2 <= CNPOR <= 1.0；若疑似百分数会自动除以 100")
    print("  常见占位值会转为 NaN:", COMMON_INVALID_VALUES)
    print("=" * 90)

    if not csv_files:
        print("没有找到 CSV 文件，请检查 CSV_DIR 路径。")
        return

    output_dir = None
    if not OVERWRITE:
        output_dir = CSV_DIR.parent / f"{CSV_DIR.name}_cleaned"

    all_reports = []
    for csv_path in csv_files:
        report = process_one_csv(csv_path, output_dir)
        all_reports.append(report)

    fixed_files = sum(1 for r in all_reports if r.get("changed"))
    failed_files = sum(1 for r in all_reports if r.get("failed"))

    total_common_invalid = {k: 0 for k in VALID_RANGES.keys()}
    total_range_invalid = {k: 0 for k in VALID_RANGES.keys()}

    for r in all_reports:
        for log_name in VALID_RANGES.keys():
            info = r.get("detail", {}).get(log_name)
            if info and info.get("status") == "ok":
                total_common_invalid[log_name] += int(info.get("common_invalid", 0))
                total_range_invalid[log_name] += int(info.get("range_invalid", 0))

    print("=" * 90)
    print("处理完成")
    print(f"发生修改的文件数: {fixed_files}")
    print(f"处理失败的文件数: {failed_files}")
    print("异常值统计汇总：")
    for log_name in VALID_RANGES.keys():
        print(
            f"  {log_name:<5} | "
            f"占位异常总数: {total_common_invalid[log_name]:<8} | "
            f"范围异常总数: {total_range_invalid[log_name]}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
