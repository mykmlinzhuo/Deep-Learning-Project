from inspiremusic.cli.inference import InspireMusicModel
from inspiremusic.cli.inference import env_variables

if __name__ == "__main__":
  env_variables()
  model = InspireMusicModel(model_name = "InspireMusic-Base")
  # just use audio prompt
  model.inference("continuation", None, "audio_prompt.wav")
  # use both text prompt and audio prompt
#   model.inference("continuation", "Continue to generate jazz music.", "audio_prompt.wav")