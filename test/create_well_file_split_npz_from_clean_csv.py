from pathlib import Path
import random
import numpy as np
import pandas as pd


# ============================================================
# 1. 修改这里
# ============================================================

# 清洗后的 CSV 文件夹
CSV_DIR = Path(r"/home/tet/zhaoheng/fast-api-project/data/2023_log_csv")

# 输出 npz 文件路径
# 建议放到你的 CDDPM 脚本中的 MODEL_DIR 下
OUT_NPZ_PATH = Path(r"/home/tet/zhaoheng/fast-api-project/src/CDDPM_RESNET_LOG/models")

# 固定随机种子，保证每次划分一致
SEED = 42

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 训练代码要求的列
REQUIRED_COLUMNS = ["Depth", "GR", "RHOB", "RILD", "CNPOR"]

# 和训练脚本保持一致
WINDOW_SIZE = 128

# 最低观测比例
# 这里是整口井四条曲线整体有效值比例
MIN_OBS_RATIO = 0.50


def find_col(df: pd.DataFrame, target: str):
    """
    忽略大小写查找列名。
    """
    col_map = {c.upper(): c for c in df.columns}
    return col_map.get(target.upper())


def is_valid_csv(csv_path: Path):
    """
    判断一口井 CSV 是否适合进入训练/测试划分。

    返回：
        valid: bool
        reason: str
    """
    try:
        df = pd.read_csv(csv_path)

        # 检查必要列
        missing_cols = []
        matched_cols = {}
        for col in REQUIRED_COLUMNS:
            matched = find_col(df, col)
            if matched is None:
                missing_cols.append(col)
            else:
                matched_cols[col] = matched

        if missing_cols:
            return False, f"missing columns: {missing_cols}"

        # Depth 有效性
        depth = pd.to_numeric(df[matched_cols["Depth"]], errors="coerce")
        valid_depth = depth.notna()

        if valid_depth.sum() < WINDOW_SIZE:
            return False, f"valid depth length < WINDOW_SIZE: {valid_depth.sum()} < {WINDOW_SIZE}"

        # 按 Depth 有效行过滤
        df2 = df.loc[valid_depth].copy()

        if len(df2) < WINDOW_SIZE:
            return False, f"rows after depth filter < WINDOW_SIZE: {len(df2)} < {WINDOW_SIZE}"

        # 四条曲线有效比例
        logs = []
        for col in ["GR", "RHOB", "RILD", "CNPOR"]:
            s = pd.to_numeric(df2[matched_cols[col]], errors="coerce")
            logs.append(s.to_numpy())

        arr = np.stack(logs, axis=1).astype(np.float32)
        obs_ratio = float(np.isfinite(arr).mean())

        if obs_ratio < MIN_OBS_RATIO:
            return False, f"obs_ratio too low: {obs_ratio:.3f} < {MIN_OBS_RATIO}"

        return True, f"ok | rows={len(df2)}, obs_ratio={obs_ratio:.3f}"

    except Exception as e:
        return False, f"read failed: {e}"


def split_files(valid_files):
    """
    按 8:1:1 随机划分。
    """
    rng = random.Random(SEED)
    files = valid_files.copy()
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    # 保证测试集至少有 1 个，如果总数足够
    if n >= 10:
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    else:
        n_test = max(1, n - n_train - n_val)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # 极小数据集兜底
    if len(test_files) == 0 and len(train_files) > 1:
        test_files = [train_files.pop()]
    if len(val_files) == 0 and len(train_files) > 2:
        val_files = [train_files.pop()]

    return train_files, val_files, test_files


def main():
    print("=" * 90)
    print(f"CSV_DIR = {CSV_DIR}")
    print(f"OUT_NPZ_PATH = {OUT_NPZ_PATH}")
    print(f"SEED = {SEED}")
    print(f"split = {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO}")
    print(f"WINDOW_SIZE = {WINDOW_SIZE}")
    print(f"MIN_OBS_RATIO = {MIN_OBS_RATIO}")
    print("=" * 90)

    csv_files = sorted(list(CSV_DIR.glob("*.csv")) + list(CSV_DIR.glob("*.CSV")))
    print(f"发现 CSV 文件数量: {len(csv_files)}")

    valid_files = []
    invalid_records = []

    for p in csv_files:
        valid, reason = is_valid_csv(p)
        if valid:
            valid_files.append(p)
            print(f"[VALID] {p.name}: {reason}")
        else:
            invalid_records.append((p.name, reason))
            print(f"[SKIP]  {p.name}: {reason}")

    if len(valid_files) < 3:
        raise RuntimeError(f"有效 CSV 文件太少，无法划分 train/val/test: {len(valid_files)}")

    train_files, val_files, test_files = split_files(valid_files)

    OUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 注意：这里只保存文件名，不保存完整路径
    # 这样和你的 load_split_files() 逻辑兼容：
    # name_map = {p.name: p for p in all_files}
    np.savez(
        OUT_NPZ_PATH,
        train=np.array([p.name for p in train_files], dtype=object),
        val=np.array([p.name for p in val_files], dtype=object),
        test=np.array([p.name for p in test_files], dtype=object),
    )

    print("=" * 90)
    print("NPZ 生成完成")
    print(f"保存路径: {OUT_NPZ_PATH}")
    print(f"train: {len(train_files)}")
    print(f"val  : {len(val_files)}")
    print(f"test : {len(test_files)}")
    print("=" * 90)

    print("train 示例:")
    for p in train_files[:10]:
        print(f"  {p.name}")

    print("val 示例:")
    for p in val_files[:10]:
        print(f"  {p.name}")

    print("test 示例:")
    for p in test_files[:10]:
        print(f"  {p.name}")

    # 保存一份跳过文件记录，方便排查
    if invalid_records:
        report_path = OUT_NPZ_PATH.with_suffix(".invalid_files.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            for name, reason in invalid_records:
                f.write(f"{name}\t{reason}\n")
        print(f"跳过文件记录已保存: {report_path}")


if __name__ == "__main__":
    main()
