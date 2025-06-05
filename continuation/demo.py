from inspiremusic.cli.inference import InspireMusicModel
from inspiremusic.cli.inference import env_variables

import os

if __name__ == "__main__":
	env_variables()
	model = InspireMusicModel(model_name = "InspireMusic-1.5B-Long", model_dir="/nvme0n1/xmy/InspireMusic-1.5B-Long")
	# just use audio prompt
	# model.inference("continuation", None, "audio_prompt.wav")
	# use both text prompt and audio prompt
	model.inference("continuation", "Continue to generate pure music with no lyrics.", "./exp/extract.flac", time_end=20.0)
	# Rename the output file
	os.rename("exp/InspireMusic-1.5B-Long/output_audio.wav", "exp/output1.wav")

	# model.inference("continuation", "Continue to generate pure music, with a soft and gentle ending.", "./exp/reversed.flac", time_end=10.0)
	# os.rename("exp/InspireMusic-1.5B-Long/output_audio.wav", "exp/output2.wav")
