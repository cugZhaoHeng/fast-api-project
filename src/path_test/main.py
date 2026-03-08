from pathlib import Path
import numpy as np


a = Path(__file__)
print(f"a: {a}, {type(a)}")
current_file = Path(__file__).resolve()
print(f"current_file:{current_file}, {type(current_file)}")
current_dir = current_file.parent
print(f"current_dir:{current_dir}, {type(current_dir)}")
project_root = current_dir.parent.parent
print(f"project_root:{project_root}, {type(project_root)}")

# data_dir = project_root /"data" / "npy_files" / "model_0000.npy"
data_dir = project_root /"data/npy_files/model_0000.npy"
print(f"data_dir:{data_dir}, {type(data_dir)}")
b = np.load(data_dir)
print(b.shape)
print(f"type b: {type(b)}")
print(f"shape b: {b.shape}")

npy_file_dir = project_root /"data/npy_files"
files = npy_file_dir.glob("*")
for file in files:
    print(f"文件: {file}")
