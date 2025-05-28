# Multitrack Generation with MusicGen

## Install
First of all, download the `musicgen-stereo-melody-large` checkpoint to local:

```bash
modelscope download --model facebook/musicgen-stereo-melody-large --local_dir <your_local_dir>
```

Then install the requirements:

```bash
pip install git+https://github.com/facebookresearch/audiocraft.git
# Or install from source
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
```

## Running
In `musicgen.py`, replace in line 5 and line 10:
```python
melody_waveform, sr = torchaudio.load("original.wav") # Change to your melody audio file

model = MusicGen.get_pretrained('/nvme0n1/xmy/musicgen-stereo-melody-large') # Change to your local model path
```

Then run `mix.py` to generate the multitrack audio, you may need to change the `AudioSegment.from_wav("original.wav")` line to your melody audio file path.

```bash
python mix.py
```
