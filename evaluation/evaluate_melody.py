import os
import numpy as np
import librosa
import crepe
import madmom
import openl3
from scipy.spatial.distance import pdist


def extract_spectral(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    flux = librosa.onset.onset_strength(y=y, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    return dict(centroid=centroid, bandwidth=bandwidth, flux=flux, flatness=flatness)

def extract_tonal(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean(axis=1)
    return dict(chroma=chroma, tonnetz=tonnetz)

def extract_pitch(y, sr):
    time, frequency, _, _ = crepe.predict(y, sr, viterbi=True) # :contentReference[oaicite:25]{index=25}
    return dict(f0_mean=frequency.mean(), f0_std=frequency.std())

def extract_rhythm(y, sr):
    proc = madmom.features.beats.RNNBeatProcessor()                  # :contentReference[oaicite:26]{index=26}
    beat_times = madmom.features.beats.DBNBeatTrackingProcessor()(proc(y))
    intervals = np.diff(beat_times)
    return dict(tempo=np.mean(60.0/intervals), tempo_var=np.std(60.0/intervals))

def extract_embedding(y, sr):
    emb, ts = openl3.get_audio_embedding(y, sr, model='mel256', input_repr='mel256')  # :contentReference[oaicite:27]{index=27}
    emb_mean = emb.mean(axis=0)
    return emb_mean

def evaluate_file(wav_path):
    """Evaluate a single WAV file and return all stats."""
    y, sr = librosa.load(wav_path, sr=None)
    print("Evaluating spec...", flush=True)
    spec = extract_spectral(y, sr)

    print("Evaluating tonal...", flush=True)
    # ton  = extract_tonal(y, sr)

    print("Evaluating pitch...", flush=True)
    pit  = extract_pitch(y, sr)

    print("Evaluating rhythm...", flush=True)
    rhy  = extract_rhythm(y, sr)

    print("Evaluating embedding...", flush=True)
    emb  = extract_embedding(y, sr)

    # Combine into one dict
    # stats = {**spec, **ton, **pit, **rhy}
    stats = {**spec, **pit, **rhy}
    stats['embedding'] = emb
    # With only one sample, diversity isn’t defined—set to 0 or None
    stats['embedding_diversity'] = 0.0
    return stats

def evaluate_directory(wav_dir):
    """Evaluate all WAVs in a directory (as before)."""
    postfix = ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.m4a', '.M4A']
    feats, embs = [], []
    for fname in os.listdir(wav_dir):
        if not any(fname.endswith(p) for p in postfix):
            continue
        path = os.path.join(wav_dir, fname)
        y, sr = librosa.load(path, sr=None)
        spec = extract_spectral(y, sr)
        ton  = extract_tonal(y, sr)
        pit  = extract_pitch(y, sr)
        rhy  = extract_rhythm(y, sr)
        emb  = extract_embedding(y, sr)
        feats.append({**spec, **ton, **pit, **rhy})
        embs.append(emb)

    emb_array = np.vstack(embs)
    diversity = pdist(emb_array, metric='cosine').mean()
    stats = {k: np.mean([f[k] for f in feats]) for k in feats[0]}
    stats['embedding_diversity'] = diversity
    return stats

def evaluate(path):
    """Wrapper: if path is a file → evaluate_file, else directory → evaluate_directory."""
    postfix = ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.m4a', '.M4A']
    if os.path.isfile(path) and path.lower().endswith(tuple(postfix)):
        return evaluate_file(path)
    else:
        return evaluate_directory(path)

if __name__ == '__main__':
    import json
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('path', help='WAV file or directory of WAVs')
    args = p.parse_args()

    results = evaluate(args.path)
    print(json.dumps(results, indent=2))
