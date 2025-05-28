from pydub import AudioSegment

# Load WAV files
original = AudioSegment.from_wav("original.wav")
accompany = AudioSegment.from_wav("generated_accompaniment.wav")

# Match lengths (optional: pad shorter one with silence)
min_len = min(len(original), len(accompany))
original = original[:min_len]
accompany = accompany[:min_len]

# Adjust volumes (original louder, accompany softer)
original = original + 3         # increase original by 3 dB
accompany = accompany - 6       # reduce accompany by 6 dB

# Mix them
combined = original.overlay(accompany)

# Export result
combined.export("mixed.wav", format="wav")
