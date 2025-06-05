import tqdm
from inspiremusic.cli.inference import InspireMusicModel
from inspiremusic.cli.inference import env_variables

import os

# dir_names = ['../DiffusionTunes/56', '../DiffusionTunes/672', '../DiffusionTunes/1008']

dir_names = ['../DiffusionTunes/1008']


if __name__ == "__main__":
    env_variables()
    model = InspireMusicModel(model_name = "InspireMusic-1.5B-Long", model_dir="/nvme0n1/xmy/InspireMusic-1.5B-Long", gpu=6)
    
    for dir_name in dir_names:
        os.makedirs(dir_name+"_cont", exist_ok=True)
        print(f"Processing directory: {dir_name}")
        wav_files = [f for f in os.listdir(dir_name) if f.endswith('.wav')]

        proc_bar = tqdm.tqdm(
            range(len(wav_files) // 2),
            unit="file",
            total=len(wav_files) // 2
        )
        for wav_file in wav_files:
            if wav_file.startswith('input'):
                continue

            proc_bar.set_description(f"Processing {wav_file}")

            wav_path = os.path.join(dir_name, wav_file)
            output_path = os.path.join(dir_name+"_cont", wav_file)
            print(f"Processing {wav_path} -> {output_path}")
            model.inference("continuation", "Generate a pure piano ending with no lyrics.", wav_path, time_end=30.0)
            os.rename("exp/InspireMusic-1.5B-Long/output_audio.wav", output_path)

            proc_bar.update(1)

        proc_bar.close()

    # model.inference("continuation", "Generate a pure piano ending with no lyrics.", "./exp/extract.flac", chorus="outro", time_end=30.0)
