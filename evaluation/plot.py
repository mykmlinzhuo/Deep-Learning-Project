import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_spectrogram(audio_path, spec_path):
    """
    Load an audio file, compute its spectrogram, and save the plot to a file.

    Parameters:
    - audio_path: Path to the input audio file.
    - spec_path: Path to save the spectrogram image (e.g., 'spectrogram.png').
    """
    # Load the audio file (y: audio time series, sr: sampling rate)
    y, sr = librosa.load(audio_path, sr=None)

    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    # Save to file and close the figure (no display)
    plt.savefig(spec_path)
    plt.close()

def save_mel_spectrogram(audio_path, mel_path, n_mels=128):
    """
    Load an audio file, compute its Mel spectrogram, and save the plot to a file.

    Parameters:
    - audio_path: Path to the input audio file.
    - mel_path: Path to save the Mel spectrogram image (e.g., 'mel_spectrogram.png').
    - n_mels: Number of Mel bands to generate (default: 128).
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    # Save to file and close the figure (no display)
    plt.savefig(mel_path)
    plt.close()

def save_stacked_mel_spectrogram(audio_paths, output_path, n_mels=128):
    """
    Load multiple audio files, compute each Mel spectrogram, and save them stacked
    vertically into one figure—using constrained_layout to avoid tight_layout warnings.

    Parameters:
    - audio_paths: List of paths to input audio files.
    - output_path: Path to save the stacked Mel spectrogram image (e.g., 'stacked_mel.png').
    - n_mels: Number of Mel bands to generate for each file (default: 128).
    """
    num_files = len(audio_paths)
    if num_files == 0:
        raise ValueError("audio_paths list is empty.")

    # Pre-compute all Mel spectrograms (in dB)
    mel_specs_db = []
    sample_rates = []
    for path in audio_paths:
        y, sr = librosa.load(path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        mel_specs_db.append(S_db)
        sample_rates.append(sr)

    # Create one figure with num_files rows, using constrained_layout
    fig, axes = plt.subplots(
        nrows=num_files, ncols=1, 
        figsize=(10, 4 * num_files),
        constrained_layout=True
    )

    # If only one file, axes will not be a list; force it into a list
    if num_files == 1:
        axes = [axes]

    # Plot each Mel spectrogram
    for idx, (S_db, sr) in enumerate(zip(mel_specs_db, sample_rates)):
        ax = axes[idx]
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        ax.set_title(f'Mel Spectrogram ({audio_paths[idx]})')

    # Create a single colorbar on the right spanning all subplots
    # Note: `pad` and `fraction` can be tweaked if colorbar is too large/small
    fig.colorbar(
        img,
        ax=axes,
        format='%+2.0f dB',
        location='right',
        fraction=0.02,
        pad=0.04
    )

    # Save figure and close
    plt.savefig(output_path)
    plt.close()

def stitch_img(img_path1, img_path2, output_path):
    """
    Stitch two images vertically and save the result.

    Parameters:
    - img_path1: Path to the first image.
    - img_path2: Path to the second image.
    - output_path: Path to save the stitched image.
    """
    img1 = plt.imread(img_path1)
    img2 = plt.imread(img_path2)

    stitched_img = np.vstack((img1, img2))
    plt.figure(figsize=(10, 8))
    plt.imshow(stitched_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    

# Example usage:
if __name__ == '__main__':
    # audio_file = '../1008/input_1.wav'
    # save_mel_spectrogram(audio_file, 'mel_spectrogram_input.png')

    save_stacked_mel_spectrogram(['../multitrack/flute_accompaniment.wav', '../multitrack/guitar_accompaniment.wav', '../1008_full/output_1.wav'], 'mel_spectrogram_flute.png')

    # stitch_img('mel_spectrogram_output.png', 'mel_spectrogram_guitar.png', 'mel_spectrogram_with_guitar.png')
