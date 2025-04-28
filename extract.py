### Code for extracting the most important 10 secs

import librosa
import numpy as np
import subprocess

input_file = "/nvme0n1/xmy/slakh250/train/Track00001/full.flac"
output_file = "/nvme0n1/xmy/slakh250/train/Track00001/extract.flac"

# Load the FLAC file (preserves original sampling rate)
y, sr = librosa.load(input_file, sr=None)  # :contentReference[oaicite:6]{index=6}

# Compute onset strength envelope (spectral flux)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # :contentReference[oaicite:7]{index=7}

# Default hop_length used by onset_strength is 512 samples
hop_length = 512

# Convert frame indices to time (seconds)
times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)  # :contentReference[oaicite:9]{index=9}

# Number of frames that span 10 seconds
window_frames = int((10 * sr) / hop_length)

# Compute cumulative onset strength over each 10-second window
cumulative = np.convolve(onset_env, np.ones(window_frames), mode='valid')

# Identify the start frame of the maximum window
best_frame = np.argmax(cumulative)
start_time = times[best_frame]               # :contentReference[oaicite:10]{index=10}
end_time = start_time + 10

print(f"Best 10s segment: {start_time:.2f}s to {end_time:.2f}s")

# Build ffmpeg command:
#  -ss <start_time>: seek to start position
#  -t  10:               duration of 10 seconds
#  -c  copy:             copy codecs (no re-encode)
cmd = [
    'ffmpeg',
    '-ss', f'{start_time:.3f}',
    '-i', input_file,
    '-t', '10',
    '-c', 'copy',
    '-y',  # overwrite output if exists
    output_file
]

subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # :contentReference[oaicite:12]{index=12}

print(f"Extracted highlight saved to {output_file}")
