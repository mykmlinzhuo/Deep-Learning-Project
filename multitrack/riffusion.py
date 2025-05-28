from diffusers import StableDiffusionPipeline
from riffusion.spectrogram_image import SpectrogramImage

# Load the Riffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("/nvme0n1/xmy/riffusion-model-v1")
pipe = pipe.to("cuda")  # use GPU if available

# Example: generate a 5-second clip of guitar to accompany a piano melody
prompt = "guitar accompaniment, backing the piano melody"
result = pipe(prompt, num_inference_steps=50)
spectrogram_img = result.images[0]

# Convert spectrogram image to audio waveform
spec = SpectrogramImage(spectrogram_img)
audio_wave = spec.to_audio()  # requires Riffusion's spectrogram utilities
with open("guitar_accompaniment.wav", "wb") as f:
    f.write(audio_wave)
