import os
from vocal2piano import vocal_to_piano

def batch_convert_vocal_to_piano(root_dirs, soundfont_path):
    """
    批量将 train/test 文件夹中以 .vocals.mp3 结尾的文件转换为对应的 .piano.mp3 文件。
    """
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".vocals.mp3"):
                    full_path = os.path.join(root, file)
                    output_path = full_path.replace(".vocals.mp3", ".piano.mp3")
                    print(f"🎵 Converting: {full_path} → {output_path}")
                    try:
                        vocal_to_piano(
                            input_mp3=full_path,
                            output_mp3=output_path,
                            soundfont=soundfont_path
                        )
                    except Exception as e:
                        print(f"❌ Failed on {full_path}: {e}")

# 示例使用
train_dir = "/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq/train"
test_dir = "/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq/test"
soundfont = "Yamaha_C3_Grand_Piano.sf2"

batch_convert_vocal_to_piano([train_dir, test_dir], soundfont)
