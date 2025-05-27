# Evaluation Metric for Diffusion-based Music Generation

## Usage

### Installation
```bash
pip install 'Cython < 3.0'
pip install -r requirements.txt
```

### Running the Evaluation
```bash
# Single file
python evaluate_melody.py generated.wav

# Directory of files
python evaluate_melody.py generated_wavs/
```


## Evaluation Metrics Overview

### Spectral Features

* **Spectral Centroid, Bandwidth, Rolloff & Flux**
  Measures of “brightness” and spectral dynamics are extracted via Librosa’s spectral feature module, offering insights into the harmonic content and energy distribution of the melody ([musicinformationretrieval.com][1]).

* **Spectral Flatness & Contrast**
  Quantifies noisiness versus tonality, with higher flatness indicating more noise‐like spectra, computed using Librosa or Essentia’s spectral descriptors ([essentia.upf.edu][8]).

### Tonal & Harmonic Features

* **Chroma Features**
  Captures pitch class distribution over time, useful for assessing harmonic richness, via `librosa.feature.chroma_stft` ([Paperspace by DigitalOcean Blog][2]).

* **Tonal Centroid (Tonnetz)**
  Projects chroma onto a six‐dimensional tonal space (perfect fifth, minor third, major third), revealing tonal movement and stability ([Paperspace by DigitalOcean Blog][2]).

### Pitch Analysis

* **Monophonic F₀ Contour (CREPE)**
  State‐of‐the‐art deep CNN tracker estimating fundamental frequency directly from waveform, ideal for single‐voice piano melodies ([GitHub][3]) ([arXiv][4]).

* **Onset & Polyphony Metrics (Madmom)**
  Onset detection and beat tracking measure rhythmic precision, while Madmom’s piano transcription algorithms estimate note counts and polyphony rates ([GitHub][5]) ([arXiv][6]).

### Rhythm & Beat

* **Beat & Tempo Consistency**
  Beat intervals and tempo variance gauge rhythmic steadiness using Madmom’s RNN‐based beat processor ([GitHub][5]).

* **Rhythmic Complexity**
  Statistical measures (e.g., inter‐onset interval entropy) reflect rhythmic diversity and regularity, computed from onset streams.

### Embedding‐Based Diversity

* **Deep Audio Embeddings (OpenL3)**
  Extracts high‐level representations; intra‐set pairwise distances quantify musical diversity and novelty across generated samples ([GitHub][7]).

### Dynamics & Complexity

* **RMS Energy & Dynamic Range**
  Root‐mean‐square frames characterize loudness variation; the ratio of max to min RMS indicates expressivity ([ravinkumar.com][9]).

* **High‐Level Complexity (Essentia)**
  Essentia provides descriptors like “event density” and “spectral complexity” to capture overall musical intricacy ([essentia.upf.edu][8]).


[1]: https://musicinformationretrieval.com/spectral_features.html?utm_source=chatgpt.com "Spectral Features - Music Information Retrieval"
[2]: https://blog.paperspace.com/music-genre-classification-using-librosa-and-pytorch/?utm_source=chatgpt.com "Music genre classification using Librosa and Tensorflow/Keras"
[3]: https://github.com/marl/crepe?utm_source=chatgpt.com "CREPE: A Convolutional REpresentation for Pitch Estimation - GitHub"
[4]: https://arxiv.org/abs/1802.06182?utm_source=chatgpt.com "CREPE: A Convolutional Representation for Pitch Estimation"
[5]: https://github.com/CPJKU/madmom?utm_source=chatgpt.com "CPJKU/madmom: Python audio and music signal processing library"
[6]: https://arxiv.org/abs/1605.07008?utm_source=chatgpt.com "madmom: a new Python Audio and Music Signal Processing Library"
[7]: https://github.com/marl/openl3?utm_source=chatgpt.com "OpenL3: Open-source deep audio and image embeddings - GitHub"
[8]: https://essentia.upf.edu/?utm_source=chatgpt.com "Homepage — Essentia 2.1-beta6-dev documentation"
[9]: https://ravinkumar.com/GenAiGuidebook/audio/audio_feature_extraction.html?utm_source=chatgpt.com "Audio Features — The GenAI Guidebook - Ravin Kumar"