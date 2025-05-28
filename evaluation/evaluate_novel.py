import os
import numpy as np
import torch
import librosa
import torch.nn.functional as F
import torchopenl3
import tqdm

def load_audio(path, sr=48000):
    """
    Load a single-channel WAV file and return (y, sr).
    Uses librosa under the hood. :contentReference[oaicite:8]{index=8}
    """
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
    return y, sr

def extract_embedding(path,
                      sr=48000,
                      input_repr="mel256",
                      content_type="music",
                      embedding_size=512,
                      device="cpu",
                      top_db=None):
    """
    Extract a single embedding vector for a WAV file using TorchOpenL3 :contentReference[oaicite:9]{index=9}.
    Returns a torch.Tensor of shape (embedding_size,).
    """
    # Load audio
    y, sr = load_audio(path, sr=sr)
    if top_db is not None:
        y, _ = librosa.effects.trim(y, top_db=top_db)
        print("Truncated audio length after trimming:", len(y) / sr, "seconds")

    # Prepare as (batch, samples)
    audio = torch.from_numpy(y).unsqueeze(0).to(device)
    # Load model
    model = torchopenl3.models.load_audio_embedding_model(
        input_repr=input_repr,
        content_type=content_type,
        embedding_size=embedding_size
    ).to(device)
    model.eval()
    with torch.no_grad():
        # Returns (embeddings, ts) where embeddings.shape = (batch, frames, dim)
        emb_frames, _ = torchopenl3.get_audio_embedding(
            audio, sr, model, center=True, hop_size=0.1
        )
        # Average over time frames to get single vector
        embedding = emb_frames.mean(dim=1).squeeze(0)
    return embedding.cpu()

def compute_novelty_score(wav_path,
                          reference_embeddings,
                          t_low=0.3,
                          t_high=0.8):
    """
    Compute novelty score [0,100] for a single WAV file.
    
    reference_embeddings: np.ndarray of shape (N_refs, D)
    t_low, t_high: similarity thresholds for mapping
    """
    # 1. Extract embedding for the input clip
    emb = extract_embedding(wav_path)
    # 2. Convert to tensor and normalize
    emb_norm = F.normalize(emb.unsqueeze(0), p=2, dim=1)  # (1, D)
    refs = torch.from_numpy(reference_embeddings)
    refs_norm = F.normalize(refs, p=2, dim=1)            # (N, D)
    # 3. Cosine similarities
    sims = torch.mm(emb_norm, refs_norm.T).squeeze(0)     # (N,)
    s_max = float(sims.max().item())
    # 4. Map to novelty
    if s_max < t_low or s_max > t_high:
        return 0.0
    novelty = (s_max - t_low) / (t_high - t_low) * 100.0
    return float(novelty)


if __name__ == "__main__":
    # pass
    # ── Example Usage ──
    # 1. Precompute reference embeddings for a folder of WAVs:
    # refs = []
    # dirs_bar = tqdm.tqdm(
    #     range(len(os.listdir(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed"))),
    #     unit="file",
    #     total=len(os.listdir(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed"))
    # )
    # for fn0 in os.listdir("/nvme0n1/xmy/maestro-v3.0.0/2004_processed"):
    #     if not os.path.isdir(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed/{fn0}"):
    #         continue
        
    #     dirs_bar.set_description(f"Processing {fn0}")
    #     for i, fn in enumerate(os.listdir(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed/{fn0}")):
    #         dirs_bar.update(1)
    #         if fn.endswith("original.wav"):
    #             emb = extract_embedding(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed/{fn0}/{fn}")
    #             refs.append(emb.numpy())
    #     dirs_bar.update(1)
    #     dirs_bar.set_postfix({"embeddings": len(refs)})
    # dirs_bar.close()

    # print("Extracted embeddings for", len(refs), "reference clips.")
    # refs = np.stack(refs, axis=0)  # shape (N, D)
    # np.save("refs_embeddings.npy", refs)

    # emb = extract_embedding(f"/nvme0n1/xmy/maestro-v3.0.0/2004_processed/0002/original.wav")
    emb1 = extract_embedding("input_672_0.wav", top_db=20)
    emb2 = extract_embedding("output_672_0.wav")
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    print("Cosine Similarity:", similarity)
    
    # 2. Compute novelty for a new clip:
    # ref_embs = np.load("refs_embeddings.npy")
    # score = compute_novelty_score("short.wav", ref_embs)
    # print(f"Novelty Score: {score:.1f}/100")
