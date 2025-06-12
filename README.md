# stage1_2: Humming-to-Full-Music Generation with ControlNet

This branch (`stage1_2`) focuses on generating 49-second full-length music from a short humming input, typically around 5 seconds (mitif segment). The core architecture is based on [Stable Audio Open 1.0](https://github.com/stability-AI/stable-audio-tools) with an integrated ControlNet module for time-varying conditioning.

## Setup

To begin, install the required dependencies. Note that `torchaudio` must be installed from the nightly channel:

```bash
pip3 install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Next, copy the template environment file and update it with your own settings:

```bash
cp .env.tmp .env
```

Example environment variables:

```bash
DIR_LOGS=/logs
DIR_DATA=/data

# Required if using wandb logger
WANDB_PROJECT=audioproject
WANDB_ENTITY=johndoe
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx
```

Then, log in to Hugging Face to enable downloading of Stable Audio Open weights:

```bash
huggingface-cli login
```

Finally, for demo support with MUSDB in `.mp3` format, make sure `libsndfile` is installed:

```bash
conda install -c conda-forge libsndfile
```


## Dataset (MAESTRO)

This project uses the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) as the main source for training and evaluation.

You have two options to prepare the dataset:

1. **Manual download + preprocessing**  
   You can download the official MAESTRO dataset from:  
   https://magenta.tensorflow.org/datasets/maestro  
   Then follow the preprocessing scripts in `scripts/preprocess_maestro/` to convert audio to the required format (44.1kHz stereo `.wav`, split into truncated-original pairs).

2. **Request preprocessed data**  
   Alternatively, you may contact the authors to obtain a preprocessed version of the dataset (already formatted for training).

Once prepared, the data should be placed under:

```
data/processed_maestro/train/
data/processed_maestro/test/
```
Each subfolder should contain paired `.truncated.wav` and `.original.wav` files per sample.


## Train

For training run
```
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-audio_large_humming python train.py exp=train_musdb_controlnet_audio_large_humming 
```

For resuming training with `checkpoint.ckpt` stored in `ckpts` run:
```
PYTHONUNBUFFERED=1 TAG=musdb-controlnet-audio_large_humming python train.py exp=train_musdb_controlnet_audio_large_humming \
+ckpt=ckpts/checkpoint.ckpt
```

## Evaluation
To run evaluation, please refer to the code provided in the `demo` branch. 

Example usage and CLI interface will be documented in future updates.

## Credits

- Evans, Z., Parker, J. D., Carr, C. J., Zukowski, Z., Taylor, J., & Pons, J. (2024). Stable Audio Open. arXiv preprint arXiv:2407.14358.
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3836-3847).