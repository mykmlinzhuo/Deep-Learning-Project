import os
import torch
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
import torchcrepe
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d


# Step 1: 读取哼唱音频
input_audio = "/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq/train/A Classic Education - NightOwl.vocals.mp3"
y, sr = librosa.load(input_audio, sr=16000)
sf.write("vocal.wav", y, sr)

# Step 2: 使用 torchcrepe 提取主旋律 F0
hop_length = int(sr / 100)  # 每帧 10ms
fmin, fmax = 50, 550
device = "cuda" if torch.cuda.is_available() else "cpu"

audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
pitch, periodicity = torchcrepe.predict(
    audio_tensor, sr, hop_length, fmin, fmax,
    model='tiny', batch_size=128, device=device, return_periodicity=True
)
pitch = pitch.squeeze().numpy()
confidence = periodicity.squeeze().numpy()

# Step 3: 置信度过滤（低置信度设为 0）
threshold = 0.3
f0 = np.where(confidence > threshold, pitch, 0.0)
print("非零音符帧数量：", np.sum(f0 > 0))

# Step 4: Gaussian 平滑 + 每 0.5s 采样一个音符
f0_smoothed = gaussian_filter1d(f0, sigma=1.5)
hop_time = 0.01
interval = int(0.5 / hop_time)  # 每 0.5 秒采样

notes = []
for i in range(0, len(f0_smoothed), interval):
    segment = f0_smoothed[i:i+interval]
    nonzero = segment[segment > 0]
    if len(nonzero) == 0:
        continue
    pitch_hz = np.median(nonzero)
    start = i * hop_time
    end = (i + interval) * hop_time
    notes.append((pitch_hz, start, end))
print("有效音符数（平滑+采样后）:", len(notes))

# Step 5: 写入 MIDI
midi = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)
for pitch_hz, start, end in notes:
    midi_num = int(np.clip(np.round(pretty_midi.hz_to_note_number(pitch_hz)), 0, 127))
    note = pretty_midi.Note(velocity=100, pitch=midi_num, start=start, end=end)
    piano.notes.append(note)
midi.instruments.append(piano)
midi_path = "melody_re.mid"
midi.write(midi_path)
print(f"✅ 已保存 MIDI 到: {midi_path}")

# Step 6: 使用 fluidsynth 渲染为 WAV（确保安装）
soundfont = "Yamaha_C3_Grand_Piano.sf2"  # 自行准备或替换路径
wav_out = "melody_re.wav"
os.system(f"fluidsynth -ni {soundfont} {midi_path} -F {wav_out} -r 44100")
if not os.path.exists(wav_out):
    raise FileNotFoundError("❌ fluidsynth 失败，未生成 melody.wav")

# Step 7: 转 MP3（使用 pydub）
mp3_out = "melody_re.mp3"
sound = AudioSegment.from_wav(wav_out)
sound.export(mp3_out, format="mp3")
print(f"🎹 完成！输出钢琴 MP3: {mp3_out}")
