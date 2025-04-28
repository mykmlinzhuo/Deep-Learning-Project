### Code for extrcating the first 10 secs

import os
import soundfile as sf
import numpy as np

example_length = 25
index = 0

# Root directory where your folders are
root_dir = "/nvme0n1/xmy/slakh2100_flac_redux/test"
save_dir = f"/nvme0n1/xmy/slakh250/test"

# Fade duration in seconds
fade_duration = 0.5  # 0.5 second fade-in and fade-out

def truncate(folder):
    folder_path = os.path.join(root_dir, folder)
    mix_path = os.path.join(folder_path, "mix.flac")
    folder_save_path = os.path.join(save_dir, folder)
    full_path = os.path.join(folder_save_path, "full.flac")
    trunc_path = os.path.join(folder_save_path, "trunc.flac")

    if os.path.isfile(mix_path):
        os.makedirs(folder_save_path, exist_ok=True)

        # Read the entire mix.flac
        audio, samplerate = sf.read(mix_path)

        # Save full version
        sf.write(full_path, audio, samplerate, format="FLAC")
        print(f"Created {full_path}")

        # Calculate samples
        samples_10s = int(10 * samplerate)
        fade_samples = int(fade_duration * samplerate)

        # Get first 10 seconds (clip if file is shorter)
        truncated = audio[:samples_10s]

        # Apply fade-in and fade-out if enough samples
        if len(truncated) >= 2 * fade_samples:
            fade_in_curve = np.linspace(0.0, 1.0, fade_samples)
            fade_out_curve = np.linspace(1.0, 0.0, fade_samples)

            if truncated.ndim == 1:
                truncated[:fade_samples] *= fade_in_curve
                truncated[-fade_samples:] *= fade_out_curve
            else:
                truncated[:fade_samples] *= fade_in_curve[:, None]
                truncated[-fade_samples:] *= fade_out_curve[:, None]

        # Save truncated+faded version
        sf.write(trunc_path, truncated, samplerate, format="FLAC")
        print(f"Created {trunc_path}")

        index += 1
        print("-" * 20 + str(index) + "-" * 20)
        if index >= example_length:
            return False
    else:
        print(f"No mix.flac found in {folder_path}")

    return True

# Loop over all subdirectories
for folder in os.listdir(root_dir):
    if not truncate(folder):
        break
