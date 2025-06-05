import os
from musicgen import add_instrument_accompaniment
from mix import mix_music

input_file = "../672_full/output_45.wav"
input_id = "output_45"
save_dir = "../cherry_mixed"
trial_num = 3
instruments = ["guitar", "violin", "flute", "drum", "cello", "bass", "saxophone", "trumpet"]

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for instrument in instruments:
        for i in range(trial_num):
            output_path = os.path.join(save_dir, f"{input_id}_{instrument}_accompaniment_{i}.wav")
            print(f"Generating accompaniment with {instrument} (trial {i+1}/{trial_num})...", flush=True)
            add_instrument_accompaniment(input_file, instrument, output_path)

            mixed_output_path = os.path.join(save_dir, f"{input_id}_{instrument}_mixed_{i}.wav")
            print(f"Mixing original with {instrument} accompaniment...")
            mix_music(input_file, output_path, mixed_output_path, strength=5, accompany_strength=-10)
            print(f"Mixed audio saved to '{mixed_output_path}'", flush=True)
