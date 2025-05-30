import os
import librosa
import numpy as np
import soundfile as sf

def concat(wav_paths, output_path, sr=48000):
    """
    Concatenate multiple WAV files into a single WAV file.
    
    Parameters:
    - wav_paths: List of paths to the WAV files to concatenate.
    - output_path: Path where the concatenated WAV file will be saved.
    - sr: Sample rate for the output WAV file.
    """
    audio_segments = []
    
    for path in wav_paths:
        y, orig_sr = librosa.load(path, sr=None, mono=True)
        if orig_sr != sr:
            print("Resampling {} from {} Hz to {} Hz".format(path, orig_sr, sr))
            y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
        audio_segments.append(y)
    
    # Concatenate all segments
    concatenated_audio = np.concatenate(audio_segments)
    
    # Save the concatenated audio to the output path
    sf.write(output_path, concatenated_audio, sr)

dir_names = ['56_cont', '672_cont', '1008_cont']

if __name__ == "__main__":
    # Example usage
    for dir_name in dir_names:
        os .makedirs(dir_name.replace("_cont", "_full"), exist_ok=True)

        wav_files = [f for f in os.listdir(dir_name) if f.endswith('.wav')]
        for file in wav_files:
            wav_paths = [os.path.join(dir_name.replace("_cont", ""), file), os.path.join(dir_name, file)]
            output_path = os.path.join(dir_name.replace("_cont", "_full"), file)
            print(f"Concatenating {wav_paths} into {output_path}")
            concat(wav_paths, output_path)