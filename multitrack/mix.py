from pydub import AudioSegment

def mix_music(original_path, accompany_path, output_path, strength=3, accompany_strength=-6):
    """
    Mix two WAV files with specified volume adjustments.
    
    original_path: Path to the original WAV file.
    accompany_path: Path to the generated accompaniment WAV file.
    output_path: Path to save the mixed output WAV file.
    strength: Volume adjustment for the original track (in dB).
    accompany_strength: Volume adjustment for the accompaniment track (in dB).
    """
    print("Mixing audio files:")
    print(f"Original: {original_path}")
    print(f"Accompaniment: {accompany_path}")
    # Load WAV files
    original = AudioSegment.from_wav(original_path)
    accompany = AudioSegment.from_wav(accompany_path)

    # Match lengths (optional: pad shorter one with silence)
    min_len = min(len(original), len(accompany))
    original = original[:min_len]
    accompany = accompany[:min_len]

    # Adjust volumes
    original = original + strength         # increase original by `strength` dB
    accompany = accompany + accompany_strength  # adjust accompaniment by `accompany_strength` dB

    # Mix them
    combined = original.overlay(accompany)

    # Export result
    combined.export(output_path, format="wav")
    print(f"Mixed audio saved to '{output_path}'")


if __name__ == "__main__":
    # Example usage
    original_path = "../1008_full/output_0.wav"  # Path to the original WAV file
    accompany_path = "guitar_accompaniment.wav"  # Path to the generated accompaniment WAV file
    output_path = "guitar_mixed.wav"  # Output path for the mixed audio

    mix_music(original_path, accompany_path, output_path, strength=3, accompany_strength=-10)
