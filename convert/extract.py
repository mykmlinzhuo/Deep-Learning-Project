import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def process_audio(file_path, output_path, duration=15.0, sr=16000, top_db=20):
    # Load audio
    y, _ = librosa.load(file_path, sr=sr)
    
    # Detect non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        print(f"No sound detected in: {file_path}")
        trimmed = np.zeros_like(y)
    else:
        # Get first non-silent index
        start_idx = intervals[0][0]
        end_idx = start_idx + int(duration * sr)

        # Clip if longer than audio
        end_idx = min(len(y), end_idx)

        # Create a full-length array filled with zeros
        trimmed = np.zeros_like(y)
        trimmed[start_idx:end_idx] = y[start_idx:end_idx]

    # Save as wav temporarily
    temp_wav = output_path.replace(".mp3", "_temp.wav")
    sf.write(temp_wav, trimmed, sr)

    # Convert wav to mp3
    audio_seg = AudioSegment.from_wav(temp_wav)
    audio_seg.export(output_path, format="mp3")
    os.remove(temp_wav)

def batch_process(root_dir):
    for split in ["train", "test"]:
        dir_path = os.path.join(root_dir, split)
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".piano.mp3"):
                full_path = os.path.join(dir_path, file_name)
                output_path = full_path.replace(".piano.mp3", ".piano_trimmed.mp3")
                process_audio(full_path, output_path)
                print(f"Processed {file_name}")

if __name__ == "__main__":
    base_dir = "/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq"
    batch_process(base_dir)
