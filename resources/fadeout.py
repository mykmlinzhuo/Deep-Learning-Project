from pydub import AudioSegment

def add_fadeout(input_path, output_path, fadeout_duration_sec):
    """
    Adds a fade-out effect to the last `fadeout_duration_sec` seconds of a WAV file.

    Args:
        input_path (str): Path to the input WAV file.
        output_path (str): Path to save the output WAV file.
        fadeout_duration_sec (float): Duration of fade-out in seconds.
    """
    # Load the audio
    audio = AudioSegment.from_wav(input_path)
    
    # Convert fadeout duration to milliseconds
    fadeout_duration_ms = int(fadeout_duration_sec * 1000)

    # Apply fade-out
    faded = audio.fade_out(duration=fadeout_duration_ms)

    # Export the result
    faded.export(output_path, format="wav")
    print("Fade-out effect added and saved to:", output_path)

# Example usage
add_fadeout("piano2.wav", "piano2.wav", 5.0)
