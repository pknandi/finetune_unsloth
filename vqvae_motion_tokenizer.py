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
    transl = motion[:, 156:159]
    transl_vel = np.zeros_like(transl)
    transl_vel[1:] = transl[1:] - transl[:-1]
    
    return np.concatenate([global_orient, body, left_hand, right_hand, transl_vel], axis=-1).astype(np.float32)

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
    def __init__(self, input_dim=159, hidden_dim=256, latent_dim=256, codebook_size=1024):
        super().__init__()
        
        # Encoder (4x temporal downsampling)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1),
        )
        
        # Production Quantizer
        self.quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=True
        )
        
        # Decoder (4x temporal upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
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
        
        self.model.train()
        print("Training VQ-VAE Codebook...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].permute(0, 2, 1).to(self.device) # (B, 159, T)
                optimizer.zero_grad()
                
                reconstructed, vq_loss, _ = self.model(x)
                
                # Reconstruction loss + Velocity loss + Quantization loss
                rec_loss = F.l1_loss(reconstructed, x)
                vel_loss = F.mse_loss(reconstructed[:, :, 1:] - reconstructed[:, :, :-1], 
                                      x[:, :, 1:] - x[:, :, :-1])
                
                loss = rec_loss + vq_loss + vel_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    def encode(self, motion: np.ndarray) -> np.ndarray:
        self.model.eval()
        T = motion.shape[0]
        # Pad to multiple of 4 for CNN
        pad_len = (4 - (T % 4)) % 4
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