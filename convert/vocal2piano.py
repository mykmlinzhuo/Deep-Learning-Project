import os
import librosa
import pretty_midi
import soundfile as sf
from pydub import AudioSegment
import torchcrepe
import torch
import numpy as np


def vocal_to_piano(input_mp3, output_mp3="melody_piano.mp3", soundfont="Yamaha_C3_Grand_Piano.sf2"):
    """
    从人声 MP3 提取旋律并转换为钢琴 MP3 伴奏。
    :param input_mp3: 输入哼唱 mp3 文件路径
    :param output_mp3: 输出钢琴 mp3 文件路径
    :param soundfont: SoundFont 文件路径（.sf2）
    """
    # Step 1: Load vocal
    y, sr = librosa.load(input_mp3, sr=16000)
    wav_path = "vocal_temp.wav"
    sf.write(wav_path, y, sr)

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # shape: (1, T)

    hop_length = int(sr / 100)
    fmin, fmax = 50, 550
    model = 'tiny'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 2: Pitch prediction
    pitch, periodicity = torchcrepe.predict(
        audio=y,
        sample_rate=sr,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        batch_size=128,
        device=device,
        return_periodicity=True
    )

    threshold = 0.5
    f0 = pitch.squeeze().numpy()
    confidence = periodicity.squeeze().numpy()
    f0 = np.where(confidence > threshold, f0, 0.0)

    # Step 3: Convert to MIDI
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    hop_time = 0.01
    note_on = None
    for i, pitch_hz in enumerate(f0):
        if pitch_hz > 0:
            try:
                midi_num = int(np.clip(librosa.hz_to_midi(pitch_hz), 0, 127))
            except:
                continue
            time_now = i * hop_time
            if note_on is None:
                note_on = (midi_num, time_now)
            elif midi_num != note_on[0]:
                note = pretty_midi.Note(velocity=100, pitch=note_on[0],
                                        start=note_on[1], end=time_now)
                piano.notes.append(note)
                note_on = (midi_num, time_now)
        elif note_on is not None:
            time_now = i * hop_time
            note = pretty_midi.Note(velocity=100, pitch=note_on[0],
                                    start=note_on[1], end=time_now)
            piano.notes.append(note)
            note_on = None

    midi.instruments.append(piano)
    midi_path = "melody_temp.mid"
    midi.write(midi_path)

    # Step 4: Convert MIDI to WAV
    wav_out = "melody_temp.wav"
    os.system(f"fluidsynth -ni {soundfont} {midi_path} -F {wav_out} -r 44100")

    # Step 5: Convert to MP3
    if not os.path.exists(wav_out):
        raise FileNotFoundError("fluidsynth 生成 wav 失败")
    sound = AudioSegment.from_wav(wav_out)
    sound.export(output_mp3, format="mp3")

    print("🎹 完成！输出文件：", output_mp3)


# 示例入口
if __name__ == "__main__":
    vocal_to_piano(
        input_mp3="/cephfs/shared/linzhuo/stable-audio-controlnet/data/musdb18hq/train/A Classic Education - NightOwl.vocals.mp3",
        output_mp3="melody_piano.mp3",
        soundfont="Yamaha_C3_Grand_Piano.sf2"
    )