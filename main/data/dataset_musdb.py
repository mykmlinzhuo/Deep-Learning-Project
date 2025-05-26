from functools import partial
from typing import List, Optional

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.autodecode import torch_audio
from torchaudio.functional import resample
import torch.nn.functional as F
import random


def _fn_resample(sample, sample_rate):
    sample_rate_orig = sample[-1]
    resampled = {stem: resample(track, orig_freq=sample_rate_orig, new_freq=sample_rate)
                 for stem, track in sample[0].items()}
    
    print(f"[RESAMPLE] Stem keys after resampling: {list(resampled.keys())}")
    return resampled, sample_rate


def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (batch, source) weights for signals that are nonzero."""
  source_norms = torch.sqrt(torch.mean(source_waveforms ** 2, dim=-1))
  return torch.greater(source_norms, 1e-8)


def _fn_extract_stems_and_pad(sample):
    max_len = max([v[0].shape[-1] for k, v in sample.items() if k.endswith(".mp3") or k.endswith(".wav")])
    default_sr = [v[1] for k, v in sample.items() if k.endswith(".mp3") or k.endswith(".wav")][0]
    # stem_to_data = ({stem + "_1": F.pad(tensor, (0, max_length - tensor.size(1))) for stem, (tensor, sr) in
    #                 stem_to_data.items()}, default_sr)
    parsed = {k.split('.')[0]: F.pad(v[0], (0, max_len - v[0].shape[-1]))
              for k, v in sample.items() if (k.endswith(".mp3") or k.endswith(".wav"))}

    print(f"[PAD] Parsed stem keys: {list(parsed.keys())}")
    return parsed, default_sr


def _get_slices(src, chunk_dur):
    for sample in src:
        stems, sr = sample
        print(f"[SLICE-IN] Received stems: {list(stems.keys())}")
        
        channels, length = list(stems.values())[0].shape
        chunk_size = int(sr * chunk_dur)

        if length < chunk_size:
             padding = torch.zeros(channels, chunk_size - length)
             stems = {stem: torch.cat([track, padding], dim=-1)
                      for stem, track in stems.items()}
             length = chunk_size

        max_shift = length - (length // chunk_size) * chunk_size
        shift = torch.randint(0, max_shift + 1, (1,)).item()

        for i in range(length // chunk_size):
            start_idx = min(length - chunk_size, i * chunk_size + shift)
            end_idx = start_idx + chunk_size
            start_s = start_idx / sr

            chunks = {stem: track[:, start_idx: end_idx] for stem, track in stems.items()}
            print(f"[SLICE] Chunk keys before RMS filter: {list(chunks.keys())}")

            chunks = {k: v for k, v in chunks.items() if _weights_for_nonzero_refs(v.sum(dim=0))}
            print(f"[SLICE] Chunk keys after RMS filter: {list(chunks.keys())}")

            if len(chunks) < 2 or (len(chunks) == 2 and "vocals" in list(chunks.keys())):
                print(f"[SLICE] Skipped chunk due to insufficient keys: {list(chunks.keys())}")
                continue

            yield chunks, start_s, length / sr


def create_musdb_dataset(
        path: str,
        sample_rate: int,
        chunk_dur: Optional[float] = None,
        shardshuffle: bool = False):

    fill_missing_keys_and_pad = partial(_fn_extract_stems_and_pad)
    get_slices = partial(_get_slices, chunk_dur=chunk_dur)
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    # create datapipeline
    dataset = (wds.WebDataset(path, shardshuffle=shardshuffle).decode(torch_audio).
               map(fill_missing_keys_and_pad).map(fn_resample))
    dataset = dataset.compose(get_slices) if chunk_dur is not None else dataset
    return dataset


def collate_fn_conditional(samples, drop_vocals=True):

    start_seconds = [x for _, x, _ in samples]
    total_seconds = [x for _, _, x in samples]
    samples = [x for x, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if 'vocals' in sample:
                sample.pop('vocals')

    subsets_in = [random.sample(list(range(len(sample))), k=random.randint(1, len(sample) - 1)) for sample in samples]
    subsets_out = [random.sample(list(set(range(len(samples[i]))) - set(indices)), k=1) for i, indices in enumerate(subsets_in)]

    outputs = []
    inputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        print(f"Sample keys: {list(sample.keys())}")
        in_indices, out_indices = subsets_in[i], subsets_out[i]
        in_stems_prompt = [stem_keys[i] for i in in_indices]
        out_stems_prompt = [stem_keys[i] for i in out_indices]
        in_track = torch.stack([sample[stem_keys[i]] for i in in_indices]).sum(dim=0, keepdim=True)
        out_track = torch.stack([sample[stem_keys[i]] for i in out_indices]).sum(dim=0, keepdim=True)
        outputs.append(out_track)
        inputs.append(in_track)
        prompts.append(f"in: {', '.join(in_stems_prompt)}; out: {', '.join(out_stems_prompt)}")

    return torch.concat(outputs), torch.concat(inputs), prompts, start_seconds, total_seconds

def collate_fn_mix(samples, drop_vocals=True):
    start_seconds = [x for _, x, _ in samples]
    total_seconds = [x for _, _, x in samples]
    samples       = [x for x, _, _ in samples]

    if drop_vocals:
        for sample in samples:
            if 'vocals' in sample:
                sample.pop('vocals')

    outputs = []
    prompts = []

    for i, sample in enumerate(samples):
        stem_keys = list(sample.keys())
        out_track = torch.stack(list(sample.values())).sum(dim=0, keepdim=True)
        outputs.append(out_track)
        prompts.append(f"out: {', '.join(stem_keys)}")

    return torch.concat(outputs), prompts, start_seconds, total_seconds

def collate_fn_piano_to_random_stem(samples):
    start_seconds = [x for _, x, _ in samples]
    total_seconds = [x for _, _, x in samples]
    samples = [x for x, _, _ in samples]

    outputs = []
    inputs = []
    prompts = []

    for i, sample in enumerate(samples):
        print(f"Sample keys: {list(sample.keys())}")
        if 'humming' not in sample:
            raise ValueError(f"Sample {i} missing 'humming' key")

        # 筛掉 piano（不是 piano_trimmed）和 piano_trimmed
        output_candidates = [k for k in sample.keys() if k not in ['humming', 'piano']]
        if not output_candidates:
            raise ValueError(f"Sample {i} has no valid output stems (only has piano_trimmed and/or piano)")

        out_key = random.choice(output_candidates)

        in_track = sample['humming'].unsqueeze(0)
        out_track = sample[out_key].unsqueeze(0)

        inputs.append(in_track)
        outputs.append(out_track)
        prompts.append(f"in: humming; out: {out_key}")

    return torch.concat(outputs), torch.concat(inputs), prompts, start_seconds, total_seconds

if __name__ == '__main__':
    dataset = create_musdb_dataset("../../data/musdb18hq/test.tar",
                                    sample_rate=44100,
                                    chunk_dur=47.57)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            pin_memory=True,
                            collate_fn=collate_fn_piano_to_random_stem,
                            num_workers=0)
    for batch in dataloader:
        print(batch)