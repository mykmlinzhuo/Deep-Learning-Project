import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Try to import music21 for key analysis; if not available, we'll skip that part
try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False

############################
# 1. Audio Loading Function #
############################

def load_audio(filepath, target_sr=None):
    """
    Load an audio file. Returns a waveform (as NumPy array) and sample rate.
    Uses torchaudio if available for consistency with PyTorch, otherwise librosa.
    """
    waveform = None
    sr = None
    try:
        # torchaudio approach (returns Tensor)
        waveform, sr = torchaudio.load(filepath)
        waveform = waveform.numpy().squeeze()  # convert to numpy array
        if waveform.ndim > 1:
            # if stereo, mix down to mono
            waveform = waveform.mean(axis=0)
        if target_sr and sr != target_sr:
            # resample to target sampling rate
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(torch.tensor(waveform)).numpy().squeeze()
            sr = target_sr
    except Exception as e:
        # fallback to librosa
        waveform, sr = librosa.load(filepath, sr=target_sr)
    return waveform, sr

####################################
# 2. Feature Extraction Functions  #
####################################

def extract_pitch_and_onsets(y, sr):
    """
    Extract fundamental pitch (F0) over time and note onset times from the waveform.
    Returns:
        pitches_hz: list of detected pitch for each note (Hz, np.nan for unvoiced segments)
        onset_times: list of onset times (in seconds) for each note onset
    """
    # Onset detection to get note start times
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    onset_times = onset_frames.tolist()
    # Use librosa's pyin to get frame-wise pitch estimates (for monophonic content)
    # We set a reasonable range for piano notes (e.g., C2 ~ 65 Hz to C7 ~ 2093 Hz)
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    f0, voiced_flags, voiced_probs = librosa.pyin(y, sr=sr, frame_length=2048, hop_length=256,
                                                 fmin=fmin, fmax=fmax)
    # Time array for each F0 frame
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    # If no onsets were detected (e.g., very simple or continuous sound), 
    # assume single note from t=0
    if len(onset_times) == 0:
        onset_times = [0.0]
    # Add the end of the piece as an "onset" to capture final note duration
    total_dur = len(y) / sr
    onset_times.append(total_dur)
    # Now, for each interval between onset_times[i] and onset_times[i+1], 
    # determine a representative pitch
    pitches_hz = []
    for i in range(len(onset_times)-1):
        start_time = onset_times[i]
        end_time = onset_times[i+1]
        # consider pitches within this interval
        mask = (times >= start_time) & (times < end_time)
        if not np.any(mask):
            # no frames in this interval (shouldn't happen if onsets cover the full duration)
            pitch_val = np.nan
        else:
            # take median of the F0 values in this note region (robust to noise/outliers)
            pitch_segment = f0[mask]
            if np.any(np.isfinite(pitch_segment)):
                pitch_val = np.nanmedian(pitch_segment)  # median of non-NaN pitches
            else:
                pitch_val = np.nan
        pitches_hz.append(pitch_val)
    # Remove the added end marker from onset_times
    onset_times = onset_times[:-1]
    return pitches_hz, onset_times

def analyze_pitch_harmony(pitches_hz):
    """
    Analyze pitch-related features: pitch classes, range, average interval, key coherence.
    Returns a dictionary of pitch/harmony metrics.
    """
    metrics = {}
    # Filter out any NaN (unvoiced) values and zero frequencies
    pitches_hz = [p for p in pitches_hz if (p is not None and np.isfinite(p) and p > 0)]
    if len(pitches_hz) == 0:
        # No pitch detected (e.g., silent input)
        metrics.update({
            "pitch_range": 0,
            "avg_interval": 0,
            "pitch_class_entropy": 0,
            "key": None,
            "key_coherence": 0
        })
        return metrics
    # Convert to MIDI notes for discrete analysis
    pitches_midi = [int(round(librosa.hz_to_midi(p))) for p in pitches_hz]
    # Pitch range in semitones
    metrics["pitch_range"] = max(pitches_midi) - min(pitches_midi) if pitches_midi else 0
    # Average interval between consecutive pitches
    intervals = [abs(pitches_midi[i+1] - pitches_midi[i]) for i in range(len(pitches_midi)-1)]
    metrics["avg_interval"] = np.mean(intervals) if intervals else 0
    # Pitch class (octave-independent) usage
    pitch_classes = [m % 12 for m in pitches_midi]
    counts = np.bincount(pitch_classes, minlength=12)
    # Normalize to probabilities for entropy
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    # Shannon entropy of pitch class distribution (measure of variety)
    entropy = -np.nansum([p * np.log2(p) for p in probs if p > 0])
    metrics["pitch_class_entropy"] = entropy
    # Key detection using music21 (if available)
    if HAS_MUSIC21:
        # Create a music21 stream from the pitches
        stream = music21.stream.Stream()
        for midi_val in pitches_midi:
            # Use quarter note duration for all notes as a placeholder (actual rhythm not needed for key)
            n = music21.note.Note(midi_val)
            stream.append(n)
        key = stream.analyze('key')  # analyze key
        metrics["key"] = str(key)  # e.g., "C major" or "a minor"
        # music21 provides a correlation attribute or scale weight for how strongly this key fits
        try:
            coherence = key.correlation  # if available
        except Exception:
            # as fallback, we can derive coherence by percentage of notes in the key's scale
            if key and hasattr(key, 'getScale'):
                scale_pitches = [p.pitchClass for p in key.getScale().getPitches(key.tonic, key.tonic.transpose(12))]
                in_scale = sum((pc in scale_pitches) for pc in pitch_classes)
                coherence = in_scale / len(pitch_classes)
            else:
                coherence = 0
        metrics["key_coherence"] = float(coherence) if coherence is not None else 0.0
    else:
        metrics["key"] = None
        metrics["key_coherence"] = 0.0
    return metrics

def analyze_rhythm(onset_times):
    """
    Analyze rhythmic features: tempo, note density, IOI variability.
    Returns a dictionary of rhythm metrics.
    """
    metrics = {}
    if len(onset_times) < 2:
        # Not enough notes to analyze rhythm (e.g., only one note)
        print("Not enough onsets detected for rhythm analysis.", flush=True)
        metrics.update({
            "note_count": len(onset_times),
            "avg_IOI": 0,
            "IOI_std": 0,
            "tempo": 0,
        })
        return metrics
    # Note count
    metrics["note_count"] = len(onset_times)
    # Compute inter-onset intervals (IOIs) in seconds
    iois = np.diff(onset_times)
    metrics["avg_IOI"] = float(np.mean(iois))
    metrics["IOI_std"] = float(np.std(iois))
    # Estimate tempo (beats per minute) using the average IOI (assuming quarter-note ~ IOI if consistent)
    # If IOI is very small, that might be multiple notes per beat; here we just do a simple estimate:
    avg_ioi = metrics["avg_IOI"]
    if avg_ioi > 0:
        bpm = 60.0 / avg_ioi  # treating avg IOI as one beat
    else:
        bpm = 0
    metrics["tempo"] = float(bpm)
    # (Alternatively, could use librosa.beat.tempo for a more robust tempo estimate.)
    return metrics

def analyze_timbre_and_audio(y, sr):
    """
    Analyze timbral and audio quality features: spectral centroid, bandwidth, noise level.
    Returns a dictionary of timbre metrics.
    """
    metrics = {}
    # Convert to torch tensor for torchaudio processing
    waveform = torch.tensor(y, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # shape (1, n_samples)
    # Compute mel spectrogram (log scale)
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=40)
    mel_spec = mel_spec_transform(waveform)  # shape: (1, n_mels, time)
    mel_spec = mel_spec[0].numpy()  # take the first channel
    # Convert to dB scale
    mel_spec_db = 10 * np.log10(np.maximum(mel_spec, 1e-10))
    # Spectral centroid (using librosa for convenience, could do manually)
    centroids = librosa.feature.spectral_centroid(S=mel_spec, sr=sr)[0]  # centroid for each frame
    metrics["spectral_centroid_mean"] = float(np.mean(centroids)) if centroids.size > 0 else 0.0
    metrics["spectral_centroid_std"] = float(np.std(centroids)) if centroids.size > 0 else 0.0
    # Spectral flatness (measure of noise-like spectra: 1 = flat noise, 0 = peaky/tonal)
    flatness = librosa.feature.spectral_flatness(S=mel_spec)
    metrics["spectral_flatness_mean"] = float(np.mean(flatness)) if flatness.size > 0 else 0.0
    # Estimate noise level: e.g., median energy in silent parts vs during notes.
    # A simple approach: take lowest 10th percentile of mel_spec_db as noise floor
    spec_values = mel_spec_db.flatten()
    if spec_values.size > 0:
        noise_floor = np.percentile(spec_values, 10)  # 10th percentile dB
        metrics["noise_floor_db"] = float(noise_floor)
    else:
        metrics["noise_floor_db"] = 0.0
    # We could also measure harmonic distortion or inharmonicity if needed.
    return metrics

def analyze_dynamics(y, sr, onset_times):
    """
    Analyze dynamics (volume) features: dynamic range, volume variability, per-note velocity.
    Returns a dictionary of dynamics metrics.
    """
    metrics = {}
    # Compute RMS energy over short windows
    frame_length = int(0.1 * sr)  # 100ms window
    hop_length = int(0.05 * sr)   # 50ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    # Dynamic range (difference between max and min RMS in dB)
    if rms_db.size > 0:
        metrics["dynamic_range_db"] = float(np.max(rms_db) - np.min(rms_db))
        metrics["rms_std_db"] = float(np.std(rms_db))
    else:
        metrics["dynamic_range_db"] = 0.0
        metrics["rms_std_db"] = 0.0
    # Estimate per-note peak amplitudes (if onset_times available)
    note_volumes = []
    for t in onset_times:
        start_idx = int(t * sr)
        # Look at a short window after the onset for peak (e.g., 50ms)
        end_idx = min(len(y), start_idx + int(0.05 * sr))
        if start_idx < len(y):
            peak_amp = np.max(np.abs(y[start_idx:end_idx]))
            note_volumes.append(peak_amp)
    if note_volumes:
        metrics["volume_variation"] = float(np.std(note_volumes))
    else:
        metrics["volume_variation"] = 0.0
    return metrics

def analyze_novelty(pitches_hz, onset_times):
    """
    Analyze novelty/creativity features: entropy of pitch and rhythm distributions, pattern variation.
    Returns a novelty score (0-100).
    """
    # Using entropy as a proxy for novelty/diversity
    novelty_score = 0.0
    # Pitch sequence entropy (we already computed pitch class entropy in pitch analysis)
    pitches_hz = [p for p in pitches_hz if p is not None and np.isfinite(p)]
    if len(pitches_hz) > 1:
        # Pitch class entropy (normalized) from 0 to 1 (log2(12) max if all 12 equally likely)
        pitch_classes = [int(round(librosa.hz_to_midi(p))) % 12 for p in pitches_hz]
        counts = np.bincount(pitch_classes, minlength=12)
        probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts)
        pitch_entropy = -np.nansum([p * np.log2(p) for p in probs if p > 0])
        pitch_entropy_norm = pitch_entropy / np.log2(12)  # normalized 0-1
    else:
        pitch_entropy_norm = 0.0
    # Rhythm entropy (entropy of IOI distribution)
    if len(onset_times) > 2:
        iois = np.diff(onset_times)
        # Bin the IOIs into categories (e.g., short, medium, long) for entropy calc
        # We'll use 5 bins between min and max IOI
        hist, bin_edges = np.histogram(iois, bins=5, density=True)
        hist_prob = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        rhythm_entropy = -np.nansum([p * np.log2(p) for p in hist_prob if p > 0])
        # Normalize by log2(N) where N is number of bins
        rhythm_entropy_norm = rhythm_entropy / np.log2(len(hist_prob)) if len(hist_prob) > 1 else 0.0
    else:
        rhythm_entropy_norm = 0.0
    # A simple combination of the two entropy measures for novelty
    novelty_score = (pitch_entropy_norm + rhythm_entropy_norm) / 2.0 * 100.0
    # Also, we might want to cap novelty if other aspects are too poor (to avoid rewarding random output).
    return novelty_score

def analyze_continuation(short_path, long_path,
                         sr=22050, hop_length=256,
                         top_db=20, threshold=0.7):
    """
    Check whether the melody in `short_path` appears in `long_path`.

    Returns:
      match (bool): True if max similarity ≥ threshold.
      best_score (float): highest average chroma cosine-similarity [0-1].
      match_time (float): time (in seconds) in long_path where best match begins.
      scores (1D array): sliding-window similarity at each frame.
    """
    # 1. Load
    y_s, _ = load_audio(short_path, target_sr=sr)
    y_l, _ = load_audio(long_path, target_sr=sr)

    # 2. Trim leading/trailing silence from short (anywhere quieter than top_db dB)
    y_s_trim, _ = librosa.effects.trim(y_s, top_db=top_db)
    print("Short audio length after trimming:", len(y_s_trim) / sr, "seconds", flush=True)

    # 3. Chroma on trimmed & full
    C_s = librosa.feature.chroma_stft(y=y_s_trim, sr=sr, hop_length=hop_length)
    C_l = librosa.feature.chroma_stft(y=y_l,    sr=sr, hop_length=hop_length)

    # 4. Normalize each frame to unit length
    def norm(C):
        n = np.linalg.norm(C, axis=0, keepdims=True) + 1e-6
        return C / n
    C_s = norm(C_s)
    C_l = norm(C_l)

    # 5. Slide & compute average cosine similarity
    Ts = C_s.shape[1]
    Tl = C_l.shape[1]
    scores = np.zeros(max(1, Tl - Ts + 1))
    for i in range(len(scores)):
        win = C_l[:, i : i+Ts]
        # dot-product per frame, then average
        scores[i] = np.mean(np.sum(C_s * win, axis=0))

    # 6. Decide match
    best = np.max(scores)
    idx  = np.argmax(scores)
    match = best >= threshold
    time_sec = idx * hop_length / sr

    return match, float(best), float(time_sec), scores

###############################################
# 3. Main evaluation function integrating all #
###############################################

def evaluate_melody(filepath):
    """
    Evaluate a piano melody WAV file and return scores and visualizations.
    """
    # Load audio
    y, sr = load_audio(filepath)
    if y is None or len(y) == 0:
        raise ValueError("Could not load audio or audio is empty.")
    # Extract pitch and onset info
    print("Extracting pitch and onset information...", flush=True)
    pitches_hz, onset_times = extract_pitch_and_onsets(y, sr)
    print(f"This audio has {len(pitches_hz)} pitch frames and {len(onset_times)} onsets.", flush=True)

    # Analyze each aspect
    print("Inspecting pitch and harmony metrics...", flush=True)
    pitch_metrics = analyze_pitch_harmony(pitches_hz)
    print("Inspecting rhythm metrics...", flush=True)
    rhythm_metrics = analyze_rhythm(onset_times)
    print("Inspecting timbre metrics...", flush=True)
    timbre_metrics = analyze_timbre_and_audio(y, sr)
    print("Inspecting dynamics metrics...", flush=True)
    dynamics_metrics = analyze_dynamics(y, sr, onset_times)
    print("Analyzing novelty...", flush=True)
    novelty_score = analyze_novelty(pitches_hz, onset_times)
    
    # For pitch/harmony, we could use key coherence (0-1) and pitch variety to determine a score.
    print("Inspecting pitch metrics...", flush=True)
    harmony_score = 0.0
    if pitch_metrics:
        # Example weighting: key coherence (50%), pitch entropy (50%)
        harmony_score = (pitch_metrics.get("key_coherence", 0) * 50.0) + \
                        (min(pitch_metrics.get("pitch_class_entropy", 0) / np.log2(12), 1.0) * 50.0)
    
    rhythm_score = 0.0
    if rhythm_metrics:
        # Example: if note_count is 0 or 1, rhythm is trivial. Otherwise consider IOI_std vs avg_IOI.
        if rhythm_metrics.get("note_count", 0) <= 1:
            rhythm_score = 0.0
        else:
            # If IOI_std is low relative to avg_IOI, rhythm is steady; some variation is good but not too much.
            consistency = 1.0 - (rhythm_metrics.get("IOI_std", 0) / (rhythm_metrics.get("avg_IOI", 0) + 1e-6))
            consistency = max(0.0, min(consistency, 1.0))
            rhythm_score = consistency * 100.0
    
    timbre_score = 0.0
    if timbre_metrics:
        # Example: lower spectral flatness (more tonal) and moderate centroid -> better
        flatness = timbre_metrics.get("spectral_flatness_mean", 0)
        # flatness 0 -> score 100, flatness 1 -> score 0 (linear for simplicity)
        timbre_score = max(0.0, 100.0 * (1.0 - flatness))
        # Also adjust based on noise floor: if noise_floor_db is very low (quiet background) that's good
        noise_floor = timbre_metrics.get("noise_floor_db", -100.0)
        # Suppose ideal noise floor ~ -60 dB or lower (no audible noise)
        noise_score = np.clip((noise_floor + 80) / 20 * 100.0, 0.0, 100.0)  # -80dB -> 0, -60dB -> 100 (just an example)
        timbre_score = 0.7 * timbre_score + 0.3 * noise_score
    
    dynamics_score = 0.0
    if dynamics_metrics:
        # If dynamic_range_db ~ 0, very flat (bad), if it's, say, 20 dB or more, good.
        drange = dynamics_metrics.get("dynamic_range_db", 0)
        dynamics_score = np.clip(drange / 20.0 * 100.0, 0.0, 100.0)  # 20 dB = 100 points, linear scaling
        # Variation in volume (too high might indicate inconsistent playing, moderate is good)
        vol_var = dynamics_metrics.get("volume_variation", 0)
        # We expect some variation, say target ~0.1. We'll penalize either extreme:
        if vol_var < 0.01:
            # almost no variation, minus points
            dynamics_score *= 0.5
        elif vol_var > 0.5:
            # very high variation (maybe one note very loud etc.), cap the score
            dynamics_score *= 0.7
            
    # Now combine all (equal weights)
    scores = {
        "harmony_pitch": harmony_score,
        "rhythm": rhythm_score,
        "timbre_sound": timbre_score,
        "dynamics": dynamics_score,
        "novelty": novelty_score,
    }
    overall = np.mean(list(scores.values()))
    scores["overall"] = overall

    # Visualization: waveform, spectrogram, pitch histogram, IOI histogram
    # We'll generate and save plots as part of evaluation (could return fig objects if needed).
    import os
    viz_dir = "evaluation_plots"
    os.makedirs(viz_dir, exist_ok=True)
    # Waveform plot
    T = np.arange(len(y)) / sr
    plt.figure(figsize=(8, 3))
    plt.plot(T, y, color='steelblue')
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "waveform.png"))
    plt.close()
    # Spectrogram (using torchaudio's MelSpectrogram already computed, but let's do a full spectrogram for visualization)
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(D_db, sr=sr, hop_length=256, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.f dB")
    plt.title("Spectrogram (Log-Frequency)")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "spectrogram.png"))
    plt.close()
    # Pitch class histogram
    plt.figure(figsize=(5, 3))
    if len(pitches_hz) > 0:
        pitch_classes = [int(round(librosa.hz_to_midi(p))) % 12 for p in pitches_hz if p > 0 and np.isfinite(p)]
        counts = np.bincount(pitch_classes, minlength=12)
        labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        plt.bar(range(12), counts, color='orange')
        plt.xticks(range(12), labels)
        plt.title("Pitch Class Distribution")
        plt.xlabel("Pitch Class")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No pitches detected", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pitch_distribution.png"))
    plt.close()
    # Inter-onset interval (IOI) histogram
    plt.figure(figsize=(5, 3))
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        plt.hist(iois, bins='auto', color='seagreen', alpha=0.7, rwidth=0.85)
        plt.title("Inter-Onset Interval Distribution")
        plt.xlabel("Interval (seconds)")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "Insufficient onsets", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "rhythm_ioi.png"))
    plt.close()

    return scores, viz_dir

if __name__ == "__main__":
    # Example usage (assuming you have a file 'melody.wav'):
    # import os
    # scores, plot_dir = evaluate_melody('recording.wav')
    # print("Scores:", scores)
    # from IPython.display import Image, display
    # for plot_name in ['waveform.png', 'spectrogram.png', 'pitch_distribution.png', 'rhythm_ioi.png']:
    #     # display(Image(filename=os.path.join(plot_dir, plot_name)))
    #     print(f"Plot saved: {os.path.join(plot_dir, plot_name)}")

    # Or to use the continuation analysis:
    match, best_score, match_time, scores = analyze_continuation('extracted.wav', 'original.wav')
    print(f"Continuation match: {match}, Best score: {best_score:.4f}, Match time: {match_time:.1f} seconds")
