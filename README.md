# Deep-Learning-Project

## Autoregressive Semantic Closure & Track-wise Orchestration Synthesis

This branch contains the third (Autoregressive Semantic Closure) and fourth (Track-wise Orchestration Synthesis) parts of our project. Also, the code for evaluation is also included in this branch.

### Part 3: Autoregressive Semantic Closure

This part is built on top of the `Inspire Music` [github repo](https://github.com/FunAudioLLM/InspireMusic), see `/continuation` for implementation details. You will need to clone the `Inspire Music` repo and download the pre-trained model weights to run this part.

```bash
git clone https://github.com/FunAudioLLM/InspireMusic.git
modelscope download --model iic/InspireMusic-1.5B-Long --local_dir ./dir
```

Then modify the corresponding paths in `/continuation/continuation_batch.py` and run the script.


### Part 4: Track-wise Orchestration Synthesis

This part is build on top of `MusicGen`, which is installed through `from audiocraft.models import MusicGen`, see `/multitrack` for implementation details. To run this script, you will need to install the `audiocraft` package and download the pre-trained model weights.

```bash
pip install audiocraft
modelscope download --model facebook/musicgen-stereo-melody-large --local_dir ./dir
```

Then modify the corresponding paths in `/multitrack/multitrack_batch.py` and run the script.


### Evaluation

See `/evaluation` for the evaluation code. The evaluation is done on the generated piano audio files. Please make sure you have the correct requirements before running the evaluatio scripts.

```bash
cd evaluation
pip install -r requirements.txt
```

Then you can run the evaluation scripts.
