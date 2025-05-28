import os
import shutil
from tqdm import tqdm

SRC_ROOT = "/root/dataset_maestro"
DST_ROOT = "/root/autodl-tmp/processed_maestro"

TRAIN_IDX = range(1, 401)  # 0001 ~ 0400
VAL_IDX = range(401, 501)  # 0401 ~ 0500

os.makedirs(os.path.join(DST_ROOT, "train"), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, "test"), exist_ok=True)

def pad(i):
    return f"{i:04d}"

for i in tqdm(range(1, 501)):
    src_dir = os.path.join(SRC_ROOT, pad(i))
    dst_dir = os.path.join(DST_ROOT, "train" if i <= 400 else "test")

    for kind in ["truncated", "original"]:
        src = os.path.join(src_dir, f"{kind}.wav")
        dst = os.path.join(dst_dir, f"{pad(i)}.{kind}.wav")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"[WARNING] Missing {src}")