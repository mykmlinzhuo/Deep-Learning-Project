import torchaudio
import os

def get_custom_metadata(info, audio):
    full_path = info["path"]
    extract_path = os.path.join(os.path.dirname(full_path), "trunc.flac")

    # Load the extract audio
    input_audio, sr = torchaudio.load(extract_path)

    # Resample if necessary (must match the main audio's sample rate)
    if "sample_rate" in info and sr != info["sample_rate"]:
        input_audio = torchaudio.transforms.Resample(sr, info["sample_rate"])(input_audio)

    return {
        "prompt": "Please generate a melody based on the input audio placed at the start.",
        "__audio__": {
            "input_audio": input_audio
        }
    }