import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class TruncToOrigWavDataset(Dataset):
    def __init__(self, path: str, sample_rate: int = 44100, max_duration_sec: float = 47.55446713):
        self.root = path
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)

        # 只提取所有前缀（即 "0001"）
        self.prefixes = sorted(
            list(set(f.split(".")[0] for f in os.listdir(path) if f.endswith(".truncated.wav")))
        )

    def _load_and_trim(self, path):
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        # ★ 保证是 stereo ★
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # ★ 强制长度匹配 sample_size × downsample ★
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad_len = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))

        return waveform

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        trunc_path = os.path.join(self.root, f"{prefix}.truncated.wav")
        orig_path = os.path.join(self.root, f"{prefix}.original.wav")

        x = self._load_and_trim(orig_path)
        y = self._load_and_trim(trunc_path)

        prompt = f"truncated to original, piano"
        total_sec = x.shape[1] / self.sample_rate
        start_sec = 0.0

        return x, y, prompt, start_sec, total_sec

    def __len__(self):
        return len(self.prefixes)


def create_truncorig_dataset(path: str, sample_rate: int = 44100):
    return TruncToOrigWavDataset(path=path, sample_rate=sample_rate)

def simple_collate_fn(batch):
    x_batch, y_batch, prompts, starts, totals = zip(*batch)
    return torch.stack(x_batch), torch.stack(y_batch), list(prompts), list(starts), list(totals)

