import torchaudio
from audiocraft.models import MusicGen


def add_instrument_accompaniment(input_path, instrument="guitar", output_path="generated_accompaniment.wav"):
    # Load the input piano melody (WAV file)
    melody_waveform, sr = torchaudio.load(input_path)
    duration = int(melody_waveform.shape[1] / sr)

    print("Input melody duration:", duration, "seconds, sample rate:", sr)

    model = MusicGen.get_pretrained('/nvme0n1/xmy/musicgen-stereo-melody-large')
    model.set_generation_params(duration=duration)

    # Define the text prompt describing the desired accompaniment
    prompt = f"a soulful {instrument}-only accompaniment in the same style as the input melody"

    # Generate the accompaniment audio (as PyTorch tensor)
    audio_output = model.generate_with_chroma([prompt], melody_waveform[None, ...], sr)

    # `audio_output` is a batch of audio waveforms. Save the first output to a file:
    torchaudio.save(output_path, audio_output[0].cpu(), sample_rate=32000)
    print(f"Generated accompaniment saved to '{output_path}'")


if __name__ == "__main__":
    # Example usage
    input_path = "../1008_full/output_0.wav"  # Path to the input piano melody
    instrument = "flute"  # Desired accompaniment instrument
    output_path = "flute_accompaniment.wav"  # Output path for generated accompaniment

    add_instrument_accompaniment(input_path, instrument, output_path)

