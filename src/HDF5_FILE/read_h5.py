from pathlib import Path

import h5py

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT_DIR / "data"

hdf5_file_path = DATA_DIR / "hdf5_file" / "JY108-7HF_stage18(1).h5"

with h5py.File(hdf5_file_path, "r") as f:

    def show(name, obj):

        print(name)

        if isinstance(obj, h5py.Dataset):
            print(" shape =", obj.shape)
            print(" dtype =", obj.dtype)

        for k, v in obj.attrs.items():
            print(" attr:", k, "=", v)

    f.visititems(show)