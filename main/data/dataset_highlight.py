import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

class TruncToOrigWavDataset(Dataset):
    def __init__(self, path: str, sample_rate: int = 44100, max_duration_sec: float = 47.55446713):
        self.root = path
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)

        # 只提取所有前缀（即 "0001"）
        self.prefixes = sorted(
            list(set(f.split(".")[0] for f in os.listdir(path) if f.endswith(".extracted.wav")))
        )

    def _load_waveform(self, path, pad_to_length=None):
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # stereo 处理
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # optional padding
        if pad_to_length is not None:
            if waveform.shape[1] > pad_to_length:
                waveform = waveform[:, :pad_to_length]
            elif waveform.shape[1] < pad_to_length:
                pad_len = pad_to_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_len))
        return waveform

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        trunc_path = os.path.join(self.root, f"{prefix}.extracted.wav")
        orig_path = os.path.join(self.root, f"{prefix}.original.wav")

        # y: input 提示片段，不做 padding，保持真实长度
        # x: 目标音频，pad 到统一长度
        y = self._load_waveform(trunc_path)
        x = self._load_waveform(orig_path, pad_to_length=self.max_samples)

        prompt = "extracted to original, piano"
        total_sec = y.shape[1] / self.sample_rate
        start_sec = 0.0

        return x, y, prompt, start_sec, total_sec

    def __len__(self):
        return len(self.prefixes)


def create_truncorig_dataset(path: str, sample_rate: int = 44100):
    return TruncToOrigWavDataset(path=path, sample_rate=sample_rate)


def collate_fn_variable_input(batch):
    x_batch, y_batch, prompts, starts, totals = zip(*batch)

    # x 是固定长度
    x_batch = torch.stack(x_batch)

    # y 是变长，先转成 (T, 2)，再 pad，再转回来 (B, 2, T_max)
    y_transposed = [y.transpose(0, 1) for y in y_batch]  # each: (T, 2)
    y_padded = pad_sequence(y_transposed, batch_first=True)  # (B, T_max, 2)
    y_padded = y_padded.transpose(1, 2)  # (B, 2, T_max)

    # input_mask: (B, T_max)
    y_lengths = [y.shape[1] for y in y_batch]
    input_mask = torch.tensor([[1] * l + [0] * (y_padded.shape[2] - l) for l in y_lengths], dtype=torch.float)

    return x_batch, y_padded, input_mask, list(prompts), list(starts), list(totals)

