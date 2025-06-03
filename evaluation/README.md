# Evaluation Metric for Single Generated Piano Melody

## Overview

Evaluating the musical quality of a *single* generated piano melody (without any reference track) requires extracting meaningful features from the audio and assessing them against musical criteria. We'll design a Python-based evaluator that analyzes various musical dimensions — **harmony (pitch content)**, **rhythm**, **timbre (sound quality)**, **dynamics**, and **novelty** — to produce both detailed sub-scores and an overall quality score. This approach is *unsupervised/self-supervised*, meaning it doesn't compare to a ground-truth recording. Instead, it uses domain knowledge and pre-trained models to judge the melody on its own merits.

We'll use **PyTorch** and **torchaudio** for neural feature extraction (e.g., computing spectrograms or leveraging pre-trained models) and other libraries like **librosa** (audio signal analysis) and **music21** (symbolic music analysis) for specific musical features. The system will generate visualizations for key aspects: the waveform, spectrogram (frequency content), pitch distribution (harmonic content), and note density or timing (rhythmic content).

**Note:** This approach draws on ideas from current research in music generation evaluation. For example, metrics like *Pitch Class Histogram (PCH)* and *Inter-Onset Interval (IOI) distributions* are used to characterize melody and rhythm. Unlike full-reference metrics that require a reference track (e.g., comparing to a real recording), our evaluator uses *reference-free* analysis. (Metrics like the Fréchet Audio Distance are also reference-free, but they rely on comparing statistics of generated audio to a large dataset of real music, which isn't applicable for a single track without a reference distribution). Instead, we will compute absolute feature-based scores for one piece.


## Implementation

After running `evaluate_melody('your_file.wav')`, you'll get a dictionary of scores and a path to the plots. For example, `scores['overall']` might be a number like 75.3 (out of 100) indicating the evaluator's judgment of the melody's quality, alongside details like `scores['harmony_pitch']`, `scores['rhythm']`, etc., each on a 0-100 scale.

### Example (Hypothetical)

Imagine we evaluate a test melody. The output might be:

```text
Scores: {
  'harmony_pitch': 80.5,
  'rhythm': 92.3,
  'timbre_sound': 75.0,
  'dynamics': 88.4,
  'novelty': 60.0,
  'overall': 79.24
}
```

This would mean the melody is strong in rhythm (perhaps steady tempo and varied note lengths), fairly good in harmony (in-key with some pitch variety), has decent but not perfect piano sound quality, good dynamic expression, but the novelty is moderate (maybe it was a bit formulaic). The overall quality is around 79/100. The generated plots would include:

* **Waveform** – showing the amplitude over time, confirming no clipping and some dynamic variation.
* **Spectrogram** – showing the piano note harmonics and revealing the frequency content (for instance, each note’s fundamental and overtone series should be visible).
* **Pitch Class Distribution** – perhaps showing most notes fall into a particular scale (e.g., C major pitches).
* **Inter-Onset Interval Distribution** – maybe showing a main peak around 0.5s (if quarter notes at 120 BPM, 0.5s apart) and some variability indicating occasional longer or shorter notes.

Each of these aspects contributes to the interpretation of the scores. By examining the plots, one can verify why the scores are what they are (e.g., a very spiky IOI histogram would explain a high rhythm consistency score, or a spectrogram with noise in high frequencies might explain a lower timbre score).


## Feature Extraction and Scoring Metrics

### Pitch and Harmony Analysis

**Melody Extraction:** We first extract the sequence of pitches (fundamental frequencies) over time from the audio. For a piano melody, which is often monophonic or has a clear dominant pitch at any time, we can use librosa's probabilistic YIN (`pyin`) algorithm to estimate the fundamental frequency (F0) for each time frame. This gives us a time series of pitches in Hz. We then convert these to musical note values (e.g., MIDI note numbers) for analysis.

**Pitch Statistics:** From the extracted notes, we evaluate:

* **Pitch Class Histogram (PCH):** distribution of pitches modulo 12 (i.e., ignoring octaves). A balanced use of multiple pitch classes (notes) indicates melodic variety, whereas very few pitch classes may indicate a simplistic or repetitive melody. We will visualize this histogram to show which notes of the scale are used.
* **Pitch Range:** the interval between the highest and lowest note (in semitones). A larger range often indicates a more expressive melody.
* **Average Interval:** the average melodic interval between consecutive notes. Extremely small intervals (e.g., a lot of repeats or stepwise motion) or extremely large jumps might both be undesirable if overused; a moderate average interval suggests a mix of stepwise motion and leaps, often musically pleasing.
* **Scale Consistency (Harmony):** We can attempt to determine if the melody fits well in a musical key. Using **music21**, we can do key analysis on the sequence of notes (e.g., Krumhansl-Schmuckler algorithm via `stream.analyze('key')`). If the melody strongly matches a particular major/minor scale, it suggests harmonic coherence. We can score this by the key certainty (how strongly the notes align to a single scale). For example, if 90% of the notes fit a C major scale, harmony score would be high; if the melody hits many random accidentals, the score would be lower (unless intentional chromaticism is expected).

Using these features, we compute a **Harmony/Pitch score**. This could be a weighted combination of submetrics: a high score if the melody uses an interesting variety of notes *but* remains mostly in-key and with a reasonable range. For instance, we might start with 100 and subtract penalties for problems (e.g., very limited pitch range or too many out-of-scale notes).

### Rhythm Analysis

**Onset Detection:** We detect note onsets (when each note or chord begins) from the audio. Librosa provides `librosa.onset.onset_detect` which picks peaks in an onset strength envelope to find timing of note attacks. This gives us a list of timestamps for notes. We can also estimate the tempo (beats per minute) from these onsets (e.g., using `librosa.beat.tempo`).

**Rhythmic Features:** From the onset sequence, we derive:

* **Note Density:** how many notes per unit time. We can compute notes per second (or per bar, if tempo is known). A very sparse melody (few notes, long pauses) might score lower in this dimension than a reasonably active melody, but too many notes could also be negative if it feels erratic. We'll visualize a time plot or histogram of inter-onset intervals. For example, an **Inter-Onset Interval (IOI) distribution** histogram shows the variability of durations between notes. A tighter clustering might indicate a steady rhythm, whereas a wide spread indicates varying note lengths (which could be either expressive or erratic).
* **Tempo Consistency:** If a clear tempo is present, does the melody mostly stick to it (e.g., notes aligning to a beat grid)? We might measure the variance of IOIs or how well onsets quantize to a constant tempo. A consistent rhythm (not too jittery) usually improves quality.
* **Rhythmic Pattern Complexity:** We could analyze if the rhythm has repetitive patterns or syncopation. This is more advanced, but a simple proxy is the entropy of the IOI distribution – higher entropy means a greater variety of rhythmic durations (which can imply novelty, covered later, or just inconsistency).

From these, we compute a **Rhythm score**. For example, a melody that maintains a steady beat and uses a mix of note lengths could score high, whereas one with either overly simplistic rhythm (e.g., all notes same length) or completely random timing might score lower.

### Timbre and Audio Quality Analysis

Since the input is a *piano* melody generated by a diffusion model, we evaluate how realistic and clean the piano sound is:

* **Spectral Characteristics:** We compute the spectrogram of the audio (using PyTorch/torchaudio). For instance, a Mel-frequency spectrogram can be obtained via `torchaudio.transforms.MelSpectrogram`, producing a time-frequency representation. From the spectrogram, we derive metrics like:

  * **Spectral Centroid:** the "brightness" of the sound – essentially the frequency center of mass of the spectrum. For a piano, the centroid will depend on played notes (higher notes -> higher centroid) but should not be extremely high (which might indicate excessive high-frequency noise). We can take the average spectral centroid over time as a feature.
  * **Spectral Bandwidth/Flatness:** measures of how spread the frequency content is and whether the spectrum is noise-like. A good piano sound should have clear harmonic peaks (low flatness) rather than a very noisy spectrum.
  * **Harmonicity:** we might attempt to measure if the audio has clear harmonic partials (which a piano note should have) vs. inharmonic or noisy components. This could be done by measuring the ratio of energy in harmonically related frequencies vs. total energy.
* **Envelope and Clarity:** We examine the waveform for issues like clipping, background noise, or unnatural artifacts. For example, a diffusion model might introduce noise or wobble. We can measure the noise floor by looking at silent portions or using a signal-to-noise ratio (comparing sound during notes vs. between notes). A higher SNR (less noise) indicates better audio quality.

These contribute to a **Timbre/Sound Quality score**. If the spectrum matches that of a typical piano (strong fundamental frequency and harmonics for each note, appropriate decay), and there's minimal noise or distortion, the score will be high. If the audio is muffled (low high-frequency content), too bright/unnatural (very high centroid or odd overtones), or noisy, the score drops.

### Dynamics Analysis

Dynamics refer to volume changes and expressiveness over time:

* **Loudness Curve:** We compute the amplitude (e.g., root-mean-square (RMS) energy) over short windows to get a loudness envelope. By examining this, we see if the melody has variations in intensity (which is musically expressive) or is flat/constant in volume.
* **Dynamic Range:** The difference between the loudest and softest sections (in dB). A larger dynamic range is generally good (it indicates the model is not just playing at one volume). However, too large might indicate some notes are too quiet or loud in an inconsistent way.
* **Note-level dynamics:** If we can detect individual note velocities (which is hard from audio alone without a reference), we could assess if the notes have natural decay and if some accents are present. In practice, we approximate this by looking at the waveform peaks for each note onset. For each detected onset, measure the peak amplitude of that note's waveform. Then see if there's variation. Some variation is good, none (all notes same loudness) might feel mechanical.

We combine these into a **Dynamics score**. A piece with a good dynamic contour (soft parts and loud parts appropriately) scores higher. If everything is flat or if there are abrupt volume jumps that make no musical sense, the score is lower.

### Novelty and Creativity Analysis

"Novelty" here tries to capture how musically novel or surprising the melody is, without being chaotic. This is tricky to quantify, but we can use a few proxies:

* **Pitch/Rhythmic Entropy:** Compute the entropy of the pitch class distribution and the rhythm (IOI) distribution. Higher entropy means more variety. For example, if the melody uses all 7 notes of a scale frequently, pitch entropy is high; if it uses only 2 notes, entropy is low. Similarly for rhythm. We expect a quality melody to have some repetition (motifs) but also some development, so moderate entropy (or a balance between repetition and surprise) is ideal.
* **Repetition vs. Variation:** We can measure if certain melodic or rhythmic patterns repeat. For instance, a very repetitive melody (low novelty) might play the same motif over and over. We could detect this by comparing sequences of intervals: if the sequence of intervals has a lot of repeats, novelty is low. On the other hand, if every interval is completely random, novelty is high but possibly too high (lack of structure). So, we might score highest when there's a mix (some repeating motif and some new material). This could be quantified with something like an **approximate entropy** or analyzing the self-similarity matrix of the melody, but in our implementation we'll likely keep it simple due to complexity.
* **Overall Uniqueness:** If we have access to a pretrained embedding (like a music embedding from an autoencoder or a model like OpenL3 or a Music Transformer), we could generate an embedding for the melody and measure how far it is from typical known melodies. However, without a reference dataset, this is hard. Instead, the system might use a *neural network trained on music* to give a "creativity" score. For example, one could use a model to predict the next note; if the actual next note is very unpredictable (low probability), the melody is novel. For our purposes, we'll stick to simpler statistics as above, due to the lack of a training corpus in this single-sample scenario.

The **Novelty score** will be high if the melody isn't a trivial or exact copy of a common nursery rhyme or a very repetitive sequence, *and* it has some structured variety. A mid-range novelty is often most pleasant (too high might sound random; too low is boring). We ensure novelty scoring doesn't conflict with harmony/rhythm (e.g., hitting random notes off-key might increase novelty entropy but decrease harmony score). The overall evaluation will consider all aspects.

### Combining Sub-scores into an Overall Score

Each dimension (Harmony, Rhythm, Timbre, Dynamics, Novelty) yields a sub-score (for example, on a scale of 0 to 10 or 0 to 100). We will normalize these and combine them for an overall quality score. A simple approach is to take a weighted average of the sub-scores. In absence of specific weighting preferences, we can weight them equally or emphasize critical aspects slightly more. For instance, **timbre** (audio fidelity) might be a prerequisite — if the audio sounds very unnatural, that could cap the overall score despite good melody/rhythm. We could implement a rule such as if timbre score is very low, overall score is heavily penalized. Otherwise, sum or average the scores.

For demonstration, we'll compute each sub-score in a 0-100 range and take a straight average as the overall score. The output of the program will clearly list the **detailed scores** per category and the **final combined score**. We will also display plots for the waveform, spectrogram, pitch class histogram, and inter-onset interval distribution so a user or researcher can visually inspect what the evaluator is seeing.
