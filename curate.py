### Code for manipulating the slakh dataset

import os
import soundfile as sf
import numpy as np
import librosa
import subprocess
from concurrent.futures import ProcessPoolExecutor
import mido

# Root directory where your folders are
root_dir = "/nvme0n1/xmy/slakh250/train"
save_dir = f"/nvme0n1/xmy/slakh250/train"

# Fade duration in seconds
fade_duration = 0.5  # 0.5 second fade-in and fade-out

def extract_midi(input_path, output_path, start_time, end_time):
    mid = mido.MidiFile(input_path)
    new_mid = mido.MidiFile()
    
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        current_time = 0
        for msg in track:
            current_time += msg.time
            if start_time <= current_time <= end_time:
                new_msg = msg.copy(time=msg.time)
                new_track.append(new_msg)
            elif current_time > end_time:
                break
        new_mid.tracks.append(new_track)
    
    new_mid.save(output_path)

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
    else:
        print(f"No mix.flac found in {folder_path}")
        return False

    return True

def extract(folder):
    folder_path = os.path.join(root_dir, folder)
    full_path = os.path.join(folder_path, "full.flac")
    folder_save_path = os.path.join(save_dir, folder)
    extract_path = os.path.join(folder_save_path, "extract.flac")

    if os.path.isfile(full_path):
        os.makedirs(folder_save_path, exist_ok=True)

        # Read the entire mix.flac
        y, sr = librosa.load(full_path, sr=None)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # :contentReference[oaicite:7]{index=7}

        # Default hop_length used by onset_strength is 512 samples
        hop_length = 512

        # Convert frame indices to time (seconds)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)  # :contentReference[oaicite:9]{index=9}

        # Number of frames that span 10 seconds
        window_frames = int((10 * sr) / hop_length)

        # Compute cumulative onset strength over each 10-second window
        cumsum = np.concatenate([[0], np.cumsum(onset_env)])                # :contentReference[oaicite:4]{index=4}
        cumulative = cumsum[window_frames:] - cumsum[:-window_frames]       # :contentReference[oaicite:5]{index=5}

        # Identify the start frame of the maximum window
        best_frame = np.argmax(cumulative)
        start_time = times[best_frame]               # :contentReference[oaicite:10]{index=10}
        end_time = start_time + 10

        print(f"Best 10s segment: {start_time:.2f}s to {end_time:.2f}s")

        cmd = [
            'ffmpeg',
            '-ss', f'{start_time:.3f}',
            '-i', full_path,
            '-t', '10',
            '-c:a', 'flac',
            '-y',
            extract_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # :contentReference[oaicite:12]{index=12}

        midi_path = os.path.join(folder_path.replace("slakh250", "slakh2100_flac_redux"), "all_src.mid")
        midi_output_path = os.path.join(folder_save_path, "extract.mid")
        extract_midi(midi_path, midi_output_path, start_time, end_time)

        print(f"Extracted highlight saved to {extract_path}")
    else:
        print(f"No mix.flac found in {folder_path}")
        return False

    return True

# Loop over all subdirectories
if __name__ == '__main__':
    folders = os.listdir(root_dir)
    with ProcessPoolExecutor() as exe:
        for keep_running in exe.map(extract, folders):
            if not keep_running:
                break
