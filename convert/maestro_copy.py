import os
import shutil
from tqdm import tqdm

SRC_ROOT = "/root/dataset_maestro"
DST_ROOT = "/root/autodl-tmp/processed_maestro_extended"

TRAIN_IDX = range(1, 451)  # 0001 ~ 0450
VAL_IDX = range(451, 501)  # 0451 ~ 0500

os.makedirs(os.path.join(DST_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, "test"), exist_ok=True)

def pad(i):
    return f"{i:04d}"

for i in tqdm(range(1, 501)):
    src_dir = os.path.join(SRC_ROOT, pad(i))
    dst_dir = os.path.join(DST_ROOT, "train" if i <= 450 else "test")

    for kind in ["extracted", "original"]:
        src = os.path.join(src_dir, f"{kind}.wav")
        dst = os.path.join(dst_dir, f"{pad(i)}.{kind}.wav")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"[WARNING] Missing {src}")