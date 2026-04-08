# motion_kmeans_tokenizer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


SMPLX_PARTS = [
    "smplx_mesh_global_orient",
    "smplx_mesh_body_pose",
    "smplx_mesh_left_hand_pose",
    "smplx_mesh_right_hand_pose",
    "smplx_mesh_transl",
    # "smplx_mesh_betas",  # usually skip for tokenization
]


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

    if len(npy_files) == 1:
        return np.load(npy_files[0], allow_pickle=False)

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
    """
    Load one SMPL-X sequence from the directory given in your CSV.

    Example:
      datasets/small/smplx/.../AVL958

    Returns:
      motion: [T, D]
      D = 159 if include_betas=False
      D = 169 if include_betas=True
    """
    motion_dir = Path(motion_dirname)
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")

    # The .npy file name usually matches the parent sequence folder name.
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

            if part == "smplx_mesh_betas" and block.shape[0] != T:
                block = np.repeat(block[:1], T, axis=0)

        blocks.append(block[:T])

    motion = np.concatenate(blocks, axis=-1)
    return motion


def preprocess_motion(motion: np.ndarray) -> np.ndarray:
    """
    Convert raw SMPL-X features into a more tokenizer-friendly representation.

    Input expected layout:
      [global_orient(3) | body_pose(63) | left_hand(45) | right_hand(45) | transl(3)]

    Output layout:
      [global_orient | body_pose | left_hand | right_hand | transl_velocity]
    """
    if motion.shape[1] < 159:
        raise ValueError(f"Expected at least 159 dims, got {motion.shape[1]}")

    global_orient = motion[:, :3]
    body = motion[:, 3:66]
    left_hand = motion[:, 66:111]
    right_hand = motion[:, 111:156]
    transl = motion[:, 156:159]

    transl_vel = np.zeros_like(transl)
    transl_vel[1:] = transl[1:] - transl[:-1]

    motion_feat = np.concatenate(
        [global_orient, body, left_hand, right_hand, transl_vel],
        axis=-1,
    )
    return motion_feat.astype(np.float32)


class Normalizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-6

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer is not fitted.")
        return (data - self.mean) / self.std

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer is not fitted.")
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str | Path) -> None:
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]


class MotionTokenizer:
    def __init__(self, n_clusters: int = 1024):
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            verbose=1,
            random_state=42,
            n_init="auto",
        )

    def fit(self, data: np.ndarray) -> None:
        self.kmeans.fit(data)

    def encode(self, motion: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(motion).astype(np.int32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.kmeans, path)

    def load(self, path: str | Path) -> None:
        self.kmeans = joblib.load(path)


def collect_dataset_from_csv(csv_path: str | Path, include_betas: bool = False) -> List[np.ndarray]:
    df = pd.read_csv(csv_path)

    if "motion_dirname" not in df.columns:
        raise ValueError("CSV must contain a 'motion_dirname' column.")

    all_data: List[np.ndarray] = []

    for i, row in df.iterrows():
        motion_dir = row["motion_dirname"]
        try:
            motion = load_smplx_sequence(motion_dir, include_betas=include_betas)
            motion = preprocess_motion(motion)
            all_data.append(motion)
        except Exception as e:
            print(f"Skipping row {i} ({motion_dir}): {e}")

    return all_data


def fit_tokenizer_from_csv(
    csv_path: str | Path,
    save_dir: str | Path,
    n_clusters: int = 1024,
    include_betas: bool = False,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from CSV...")
    sequences = collect_dataset_from_csv(csv_path, include_betas=include_betas)

    if not sequences:
        raise RuntimeError("No valid motion sequences were loaded from the CSV.")

    print("Concatenating frames...")
    data = np.concatenate(sequences, axis=0)
    print(f"Collected {data.shape[0]} frames with dim {data.shape[1]}")

    print("Fitting normalizer...")
    norm = Normalizer()
    norm.fit(data)
    data_norm = norm.transform(data)

    print("Training KMeans tokenizer...")
    tokenizer = MotionTokenizer(n_clusters=n_clusters)
    tokenizer.fit(data_norm)

    print("Saving artifacts...")
    tokenizer.save(save_dir / "tokenizer.pkl")
    norm.save(save_dir / "normalizer.npz")

    print(f"Saved tokenizer to {save_dir / 'tokenizer.pkl'}")
    print(f"Saved normalizer to {save_dir / 'normalizer.npz'}")


def tokenize_motion_directory(
    motion_dirname: str | Path,
    tokenizer_path: str | Path,
    normalizer_path: str | Path,
    include_betas: bool = False,
) -> np.ndarray:
    tokenizer = MotionTokenizer()
    tokenizer.load(tokenizer_path)

    norm = Normalizer()
    norm.load(normalizer_path)

    motion = load_smplx_sequence(motion_dirname, include_betas=include_betas)
    motion = preprocess_motion(motion)
    motion = norm.transform(motion)

    tokens = tokenizer.encode(motion)
    return tokens


def tokens_to_text(tokens: np.ndarray) -> str:
    return " ".join(f"<m_{int(t)}>" for t in tokens)


def tokenize_csv_to_jsonl(
    csv_path: str | Path,
    tokenizer_path: str | Path,
    normalizer_path: str | Path,
    output_jsonl: str | Path,
    include_betas: bool = False,
) -> None:
    df = pd.read_csv(csv_path)
    if "audio_filename" not in df.columns or "motion_dirname" not in df.columns:
        raise ValueError("CSV must contain 'audio_filename' and 'motion_dirname' columns.")

    tokenizer = MotionTokenizer()
    tokenizer.load(tokenizer_path)

    norm = Normalizer()
    norm.load(normalizer_path)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            audio_path = row["audio_filename"]
            motion_dir = row["motion_dirname"]

            try:
                motion = load_smplx_sequence(motion_dir, include_betas=include_betas)
                motion = preprocess_motion(motion)
                motion = norm.transform(motion)
                tokens = tokenizer.encode(motion)

                sample = {
                    "id": str(i),
                    "audio_filename": audio_path,
                    "motion_dirname": motion_dir,
                    "motion_tokens": tokens_to_text(tokens),
                }
                f.write(pd.Series(sample).to_json(force_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"Skipping row {i} ({motion_dir}): {e}")

    print(f"Wrote {written} samples to {output_jsonl}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=1024)
    parser.add_argument("--include_betas", action="store_true")
    parser.add_argument("--tokenize_jsonl", action="store_true")
    parser.add_argument("--output_jsonl", type=str, default="train.jsonl")
    args = parser.parse_args()

    fit_tokenizer_from_csv(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        n_clusters=args.n_clusters,
        include_betas=args.include_betas,
    )

    if args.tokenize_jsonl:
        tokenize_csv_to_jsonl(
            csv_path=args.csv_path,
            tokenizer_path=Path(args.save_dir) / "tokenizer.pkl",
            normalizer_path=Path(args.save_dir) / "normalizer.npz",
            output_jsonl=args.output_jsonl,
            include_betas=args.include_betas,
        )