import soundfile as sf
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

def smooth(input_path, output_path):
    data, sr = sf.read(input_path)

    def lowpass_filter(audio, sr, cutoff=3000, order=5):
        nyquist = sr / 2
        norm_cutoff = cutoff / nyquist
        b, a = butter(order, norm_cutoff, btype='low')
        return filtfilt(b, a, audio, axis=0)
    
    data = lowpass_filter(data, sr)

    sf.write(output_path, data, sr)

def reverse(input_path, output_path):
    # Load the FLAC file
    data, samplerate = sf.read(input_path)  # data is a NumPy array

    # Reverse the audio
    reversed_data = data[::-1]

    # Save as FLAC
    sf.write(output_path, reversed_data, samplerate)

def extract(input_path, output_path, start_time=None, end_time=None):
    # Load audio
    data, samplerate = sf.read(input_path)

    # Trim the first second
    if end_time is None:
        end_time = data.shape[0] / samplerate
    if start_time is None:
        start_time = 0
    trimmed = data[int(samplerate * start_time): int(samplerate * end_time)]

    # Save the result
    sf.write(output_path, trimmed, samplerate)


def concatenate(path1, path2, output_path):
    # Load first file (librosa automatically converts to float32)
    data1, rate1 = librosa.load(path1, sr=None)
    data2, rate2 = librosa.load(path2, sr=None)

    # If needed, resample data2 to match rate1
    if rate1 != rate2:
        data2 = librosa.resample(data2, orig_sr=rate2, target_sr=rate1)

    # Ensure both are 2D if stereo
    if data1.ndim == 1:
        data1 = np.expand_dims(data1, axis=1)
    if data2.ndim == 1:
        data2 = np.expand_dims(data2, axis=1)

    # If necessary, pad mono to stereo (optional)
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Channel mismatch after resampling")

    # Concatenate and save
    concatenated = np.concatenate((data1, data2), axis=0)
    sf.write(output_path, concatenated, rate1)


if __name__ == "__main__":
    # extract("exp/extract.flac", "exp/extract.flac", 1.0)
    # reverse("exp/extract.flac", "exp/reversed.flac")
    # smooth("exp/reversed.flac", "exp/reversed.flac")

    reverse("exp/output2.wav", "exp/start.wav")
    # smooth("exp/start.wav", "exp/start.wav")

    concatenate("exp/start.wav", "exp/extract.flac", "exp/conc1.wav")
    concatenate("exp/conc1.wav", "exp/output1.wav", "exp/final.wav")
