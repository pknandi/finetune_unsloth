from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vector_quantize_pytorch import VectorQuantize

SMPLX_PARTS = [
    "smplx_mesh_global_orient",
    "smplx_mesh_body_pose",
    "smplx_mesh_left_hand_pose",
    "smplx_mesh_right_hand_pose",
    "smplx_mesh_transl",
]

# ==========================================
# 1. SMPL-X Loading & Preprocessing
# ==========================================
def _load_single_npy_from_folder(folder: Path, expected_stem: Optional[str] = None) -> np.ndarray:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    npy_files = sorted(folder.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {folder}")
    if expected_stem is not None:
        for f in npy_files:
            if f.stem == expected_stem:
                return np.load(f, allow_pickle=False)
    return np.load(npy_files[0], allow_pickle=False)

def _ensure_2d_frame_major(arr: np.ndarray, seq_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.repeat(arr[None, :], seq_len, axis=0)
    if arr.shape[0] == seq_len:
        return arr.reshape(seq_len, -1)
    if arr.shape[0] == 1:
        return np.repeat(arr, seq_len, axis=0).reshape(seq_len, -1)
    raise ValueError(f"Cannot align array with shape {arr.shape} to seq_len={seq_len}")

def load_smplx_sequence(motion_dirname: str | Path, include_betas: bool = False) -> np.ndarray:
    motion_dir = Path(motion_dirname)
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    expected_stem = motion_dir.parent.name
    parts = list(SMPLX_PARTS)
    if include_betas:
        parts.append("smplx_mesh_betas")

    arrays: Dict[str, np.ndarray] = {}
    for part in parts:
        part_dir = motion_dir / part
        if part_dir.exists():
            arrays[part] = _load_single_npy_from_folder(part_dir, expected_stem=expected_stem)

    if not arrays:
        raise FileNotFoundError(f"No SMPL-X parts found in {motion_dir}")

    lengths = [arr.shape[0] for arr in arrays.values() if arr.ndim >= 1]
    T = min(lengths)

    blocks: List[np.ndarray] = []
    for part in parts:
        arr = arrays.get(part)
        if arr is None:
            if part == "smplx_mesh_global_orient":
                block = np.zeros((T, 3), dtype=np.float32)
            elif part == "smplx_mesh_body_pose":
                block = np.zeros((T, 63), dtype=np.float32)
            elif part in ("smplx_mesh_left_hand_pose", "smplx_mesh_right_hand_pose"):
                block = np.zeros((T, 45), dtype=np.float32)
            elif part == "smplx_mesh_transl":
                block = np.zeros((T, 3), dtype=np.float32)
            elif part == "smplx_mesh_betas":
                block = np.zeros((T, 10), dtype=np.float32)
            else:
                continue
        else:
            block = _ensure_2d_frame_major(arr, T).astype(np.float32)
        blocks.append(block[:T])

    return np.concatenate(blocks, axis=-1)

def preprocess_motion(motion: np.ndarray) -> np.ndarray:
    global_orient = motion[:, :3]
    body = motion[:, 3:66]
    left_hand = motion[:, 66:111]
    right_hand = motion[:, 111:156]
    # Absolute world position, NOT velocity. Velocity + cumsum-at-decode was tried
    # twice and both variants fail: (a) velocity alone loses the start position
    # entirely (everything reconstructs at the origin); (b) stuffing the absolute
    # start into frame 0 of the velocity channel makes that frame a ±60-85 sigma
    # outlier after normalization — unrepresentable by the codebook — AND cumsum
    # accumulates per-frame reconstruction error into metres of drift over a clip.
    # Absolute position is bounded, normalizes cleanly, and decodes drift-free.
    transl = motion[:, 156:159]

    # global_orient is a large-swing rotation that can exceed +/-pi in raw mocap
    # data (real wrap-around observed in practice). A plain L1 loss on the raw
    # radian value treats +pi and -pi — nearly the same rotation — as maximally
    # far apart, which trains the VQ-VAE to reconstruct the wrong side of the
    # wrap. sin/cos is continuous across that boundary, so nearby rotations stay
    # numerically close no matter which side of +/-pi they land on.
    global_orient_sincos = np.concatenate([np.sin(global_orient), np.cos(global_orient)], axis=-1)

    return np.concatenate(
        [global_orient_sincos, body, left_hand, right_hand, transl], axis=-1
    ).astype(np.float32)

class Normalizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-6

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str | Path) -> None:
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]

# ==========================================
# 2. VQ-VAE Architecture
# ==========================================
class MotionVQVAE(nn.Module):
    def __init__(self, input_dim=162, hidden_dim=256, latent_dim=256, codebook_size=1024):
        super().__init__()
        
        # Encoder (2x temporal downsampling). Was 4x: each token covered an 8-16fps-
        # equivalent chunk, forcing a brief dynamic burst (e.g. starting to walk) to
        # share a handful of coarse blocks with the surrounding static motion. 2x
        # gives dynamic bursts more dedicated codes/tokens to be represented with.
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1),
        )
        
        # Production Quantizer. Euclidean distance, NOT cosine: cosine similarity
        # normalizes latents to the unit sphere, so codes capture direction only
        # and magnitude is discarded — a big arm swing and a small one collapse to
        # the same code. Measured effect: reconstructed amplitude shrank to ~0.45x
        # GT on global_orient. Euclidean codes keep magnitude, which is what
        # motion needs (and what standard motion-VQ setups use).
        self.quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=False,
            kmeans_init=True,            # seed codes from real encoder outputs, not random init
            kmeans_iters=10,
            threshold_ema_dead_code=2,    # reset codes that stop getting used instead of leaving them dead
        )
        
        # Decoder (2x temporal upsampling, matching the encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # Runs at full framerate and spans across token-block boundaries,
            # giving the decoder room to blend adjacent blocks instead of
            # concatenating independently-decoded blocks raw.
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x).permute(0, 2, 1)
        quantized, indices, vq_loss = self.quantizer(encoded)
        decoded = self.decoder(quantized.permute(0, 2, 1))
        return decoded, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x):
        encoded = self.encoder(x).permute(0, 2, 1)
        _, indices, _ = self.quantizer(encoded)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, token_ids):
        quantized = self.quantizer.get_output_from_indices(token_ids)
        return self.decoder(quantized.permute(0, 2, 1))

# ==========================================
# 3. Tokenizer Interface Wrapper
# ==========================================
class VQVAETokenizer:
    def __init__(self, n_clusters: int = 1024, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = MotionVQVAE(codebook_size=n_clusters).to(self.device)

    def fit(self, chunks: List[np.ndarray], epochs: int = 500, batch_size: int = 16) -> None:
        dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(chunks), dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)

        # Translation is only the last 3 of ~162 channels. Under a plain, unweighted
        # L1 sum it gets numerically drowned out by the other ~159 pose/hand
        # channels, so rare-but-real events (e.g. starting to walk) get averaged
        # away in favor of the dominant near-static frames. Upweight it so its
        # gradient contribution is comparable to the rest of the feature vector.
        input_dim = next(self.model.decoder[-1].parameters()).shape[0]
        loss_weights = torch.ones(input_dim, device=self.device)
        loss_weights[-3:] = 8.0
        loss_weights = loss_weights.view(1, -1, 1)

        self.model.train()
        print("Training VQ-VAE Codebook...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].permute(0, 2, 1).to(self.device) # (B, input_dim, T)
                optimizer.zero_grad()

                reconstructed, vq_loss, _ = self.model(x)

                # Long static/slow stretches vastly outnumber brief dynamic bursts
                # (e.g. starting to walk) within a clip. An unweighted per-frame
                # average lets the optimizer spend the codebook's limited capacity
                # on the dominant static regime and smooth the rare dynamic frames
                # away. Weight each frame by its own GT motion energy (relative to
                # the chunk's mean) so dynamic frames aren't averaged out.
                with torch.no_grad():
                    frame_energy = torch.zeros(x.shape[0], 1, x.shape[2], device=self.device)
                    frame_energy[:, :, 1:] = x[:, :, 1:].sub(x[:, :, :-1]).abs().mean(dim=1, keepdim=True)
                    frame_energy[:, :, 0] = frame_energy[:, :, 1]
                    frame_weight = 1.0 + 5.0 * frame_energy / (frame_energy.mean(dim=2, keepdim=True) + 1e-6)

                # Reconstruction loss (per-part + per-frame weighted) + Velocity loss + Quantization loss
                rec_loss = (loss_weights * frame_weight * (reconstructed - x).abs()).mean()
                # Velocity loss gets the same frame AND channel weighting. The
                # channel weights matter here: the normalizer's transl std is
                # dominated by between-subject studio positions (~1m), so real
                # per-frame root velocity is minuscule in normalized units and an
                # unweighted velocity term can't suppress quantization noise on
                # the root — which shows up as the character jittering/gliding
                # around its true position.
                vel_err = (reconstructed[:, :, 1:] - reconstructed[:, :, :-1]) - (x[:, :, 1:] - x[:, :, :-1])
                vel_loss = (loss_weights * frame_weight[:, :, 1:] * vel_err.pow(2)).mean()

                loss = rec_loss + vq_loss + vel_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    def encode(self, motion: np.ndarray) -> np.ndarray:
        self.model.eval()
        T = motion.shape[0]
        # Pad to multiple of 2 for CNN (matches the encoder's 2x downsampling)
        pad_len = (2 - (T % 2)) % 2
        if pad_len > 0:
            motion = np.pad(motion, ((0, pad_len), (0, 0)), mode='edge')
            
        x = torch.tensor(motion, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(self.device)
        with torch.no_grad():
            tokens = self.model.encode_to_tokens(x)
        return tokens.squeeze().cpu().numpy().astype(np.int32)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self.model.eval()
        t_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            motion = self.model.decode_from_tokens(t_tensor)
        # Safely squeeze ONLY the batch dimension
        return motion.squeeze(0).permute(1, 0).cpu().numpy()

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ==========================================
# 4. Dataset Processing & Training Loop
# ==========================================
def collect_dataset_from_csv(csv_path: str | Path) -> List[np.ndarray]:
    df = pd.read_csv(csv_path)
    all_data = []
    for i, row in df.iterrows():
        try:
            motion = load_smplx_sequence(row["motion_dirname"])
            all_data.append(preprocess_motion(motion))
        except Exception as e:
            print(f"Skipping row {i}: {e}")
    return all_data

def fit_tokenizer_from_csv(csv_path: str | Path, save_dir: str | Path, n_clusters: int = 1024) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    sequences = collect_dataset_from_csv(csv_path)
    
    print("Fitting normalizer...")
    norm = Normalizer()
    norm.fit(np.concatenate(sequences, axis=0))
    
    # Extract 120-frame (4s) overlapping windows for stable CNN training
    window_size = 120
    chunks = []
    for seq in sequences:
        seq = norm.transform(seq)
        T = seq.shape[0]
        if T < window_size:
            pad_len = window_size - T
            chunks.append(np.pad(seq, ((0, pad_len), (0, 0)), mode='edge'))
        else:
            for i in range(0, T - window_size + 1, 30):
                chunks.append(seq[i:i+window_size])

    # Clips are dominated by static/slow windows; brief dynamic bursts (walking,
    # large turns) land in only a few windows and get outvoted during training
    # even with per-frame loss weighting. Replicate high-energy windows so the
    # dynamic regime is properly represented in every epoch's batches.
    energies = np.array([np.abs(np.diff(c, axis=0)).mean() for c in chunks])
    threshold = np.median(energies) * 1.5
    dynamic_chunks = [c for c, e in zip(chunks, energies) if e > threshold]
    chunks = chunks + dynamic_chunks * 2
    print(f"Windows: {len(energies)} base + {len(dynamic_chunks)}x2 high-energy oversamples")

    tokenizer = VQVAETokenizer(n_clusters=n_clusters)
    tokenizer.fit(chunks)

    tokenizer.save(save_dir / "tokenizer.pt")
    norm.save(save_dir / "normalizer.npz")
    print(f"Saved to {save_dir}")

def tokenize_csv_to_jsonl(csv_path: str | Path, save_dir: str | Path, output_jsonl: str | Path) -> None:
    df = pd.read_csv(csv_path)
    tokenizer = VQVAETokenizer()
    tokenizer.load(Path(save_dir) / "tokenizer.pt")
    norm = Normalizer()
    norm.load(Path(save_dir) / "normalizer.npz")

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            try:
                motion = preprocess_motion(load_smplx_sequence(row["motion_dirname"]))
                motion = norm.transform(motion)
                tokens = tokenizer.encode(motion)
                
                sample = {
                    "id": str(i),
                    "audio_filename": row["audio_filename"],
                    "motion_dirname": row["motion_dirname"],
                    "motion_tokens": "".join(f"<m_{int(t)}>" for t in tokens.reshape(-1)),
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="motion_tokenizer_artifacts")
    parser.add_argument("--n_clusters", type=int, default=1024)
    parser.add_argument("--tokenize_jsonl", action="store_true")
    parser.add_argument("--output_jsonl", type=str, default="datasets/tokenized_data.jsonl")
    args = parser.parse_args()

    fit_tokenizer_from_csv(args.csv_path, args.save_dir, args.n_clusters)
    
    if args.tokenize_jsonl:
        tokenize_csv_to_jsonl(args.csv_path, args.save_dir, args.output_jsonl)