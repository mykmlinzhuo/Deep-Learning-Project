# Deep-Learning-Project (xmy branch)

This branch aims to tune stable-audio-1.0 on music-to-music generation task. 

Please first clone the github repo and install the requirements:
```bash
git clone https://github.com/Stability-AI/stable-audio-tools.git
pip install -r requirements.txt

cd stable-audio-tools
pip install -e .
```

Then, you can run the ffmpeg installation if you can not install an app:
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
