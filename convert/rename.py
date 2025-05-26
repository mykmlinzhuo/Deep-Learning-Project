import os
from glob import glob

base_dir = "/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq"

for split in ["train", "test"]:
    split_dir = os.path.join(base_dir, split)
    files = glob(os.path.join(split_dir, "*.piano_trimmed.mp3"))

    for old_path in files:
        new_path = old_path.replace(".piano_trimmed.mp3", ".humming.mp3")
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
