import torchaudio
from audiocraft.models import MusicGen

# Load the input piano melody (WAV file)
melody_waveform, sr = torchaudio.load("original.wav")
duration = int(melody_waveform.shape[1] / sr)

print("Input melody duration:", duration, "seconds, sample rate:", sr)

model = MusicGen.get_pretrained('/nvme0n1/xmy/musicgen-stereo-melody-large')
model.set_generation_params(duration=duration)

# Define the text prompt describing the desired accompaniment
prompt = "a soulful guitar-only accompaniment in the same style as the input melody"

# Generate the accompaniment audio (as PyTorch tensor)
audio_output = model.generate_with_chroma([prompt], melody_waveform[None, ...], sr)

# `audio_output` is a batch of audio waveforms. Save the first output to a file:
torchaudio.save("generated_accompaniment.wav", audio_output[0].cpu(), sample_rate=32000)
print("Generated accompaniment saved to 'generated_accompaniment.wav'")
