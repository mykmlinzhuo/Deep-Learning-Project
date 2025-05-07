# Deep-Learning-Project (xmy branch)

This branch aims to tune Audio LDM and stable-audio-1.0(**Failed**) on music-to-music generation task. 


## Audio LDM setup
Please first clone the github repo to the local dir (no need if you cloned it from this branch already):
```bash
git clone https://github.com/haoheliu/AudioLDM-training-finetuning.git
mv AudioLDM-training-finetuning AudioLDM
cd AudioLDM
```

Then, install poetry module using Pypi:
```bash
conda create -n audioldm python=3.10 -y
conda activate audioldm
pip install poetry
```
Next we need to change the source of poetry, as the original source is really slow:
```bash
poetry source add --priority=primary mirrors https://pypi.tuna.tsinghua.edu.cn/simple/
poetry lock # This might take a minute or so
```

Finally, run:
```bash
poetry install
```
and everything will be set up.


## Stable Audio setup
Please first clone the github repo and install the requirements:
```bash
git clone https://github.com/Stability-AI/stable-audio-tools.git
pip install -r requirements.txt

cd stable-audio-tools
pip install -e .
```


## FFMPEG setup
You can run the ffmpeg installation if you can not install an app:
```bash
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg
cd ffmpeg

./configure --prefix="$HOME/ffmpeg_build" --enable-shared --disable-static --disable-x86asm
make -j$(nproc)   # compile using all CPU cores
make install

# Add the ffmpeg build to your PATH
export PATH="$HOME/ffmpeg_build/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ffmpeg_build/lib:$LD_LIBRARY_PATH"
```
