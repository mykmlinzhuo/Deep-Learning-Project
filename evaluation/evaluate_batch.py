import os
import tqdm
import numpy as np

from evaluate_melody import analyze_continuation, evaluate_melody

# dir_names = ['56', '672', '1008']
dir_names = ['672_mixed']

if __name__ == "__main__":
    for dir_name in dir_names:
        # scores = []
        scores = {
            "harmony_pitch": [],
            "rhythm": [],
            "timbre_sound": [],
            "dynamics": [],
            "novelty": [],
        }
        print(f"Processing directory: {dir_name}")
        wav_files = [f for f in os.listdir(dir_name) if f.endswith('.wav')]

        proc_bar = tqdm.tqdm(
            range(len(wav_files) // 2),
            unit="file",
            total=len(wav_files) // 2
        )
        for wav_file in wav_files:
            if (not wav_file.endswith('.wav')) or wav_file.startswith('input'):
                continue

            wav_path_input = os.path.join(dir_name, wav_file.replace('output', 'input'))
            wav_path_output = os.path.join(dir_name, wav_file)
            proc_bar.set_description(f"Processing {wav_file}")
            
            
            # _, best_score, _, _ = analyze_continuation(
            #     wav_path_input,
            #     wav_path_output
            # )
            # scores.append(best_score)

            score = evaluate_melody(
                wav_path_output,
                plots=False,
                verbose=False
            )
            for key in scores.keys():
                if key in score:
                    scores[key].append(score[key])
                else:
                    scores[key].append(None)

            proc_bar.update(1)

        proc_bar.close()
        # print(f"Average score for directory {dir_name}: {np.mean(scores)}")
        print(f"Average score for directory {dir_name}:")
        for key, value in scores.items():
            if value:
                print(f"{key}: {np.mean(value)}")
            else:
                print(f"{key}: No scores available")

