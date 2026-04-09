# speech_to_motion_pipeline.py

from __future__ import annotations

# 1. FIXED: Unsloth imported FIRST to apply optimizations and avoid the warning
from unsloth import FastLanguageModel

import soundfile as sf

import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio

# 2. FIXED: Imported from standalone 'encodec' instead of 'audiocraft'
from encodec import EncodecModel
from encodec.utils import convert_audio

from sklearn.cluster import MiniBatchKMeans
from transformers import TrainingArguments, Trainer, default_data_collator


# =========================
# 1) SMPL-X motion loading
# =========================

SMPLX_PARTS = [
    "smplx_mesh_global_orient",
    "smplx_mesh_body_pose",
    "smplx_mesh_left_hand_pose",
    "smplx_mesh_right_hand_pose",
    "smplx_mesh_transl",
]


def _load_single_npy_from_folder(folder: Path, expected_stem: Optional[str] = None) -> np.ndarray:
    npy_files = sorted(folder.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder}")

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
    raise ValueError(f"Cannot align array {arr.shape} to length {seq_len}")


def load_smplx_sequence(motion_dirname: str | Path, include_betas: bool = False) -> np.ndarray:
    motion_dir = Path(motion_dirname)
    expected_stem = motion_dir.parent.name

    parts = list(SMPLX_PARTS)
    if include_betas:
        parts.append("smplx_mesh_betas")

    arrays = {}
    for part in parts:
        part_dir = motion_dir / part
        if part_dir.exists():
            arrays[part] = _load_single_npy_from_folder(part_dir, expected_stem=expected_stem)

    if not arrays:
        raise FileNotFoundError(f"No SMPL-X files found in {motion_dir}")

    T = min(arr.shape[0] for arr in arrays.values())
    blocks = []

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


# =========================
# 2) Motion tokenizer
# =========================

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-6

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted.")
        return (data - self.mean) / self.std

    def save(self, path: str | Path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str | Path):
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]


class MotionTokenizer:
    def __init__(self, n_clusters: int = 1024):
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            random_state=42,
            n_init="auto",
            verbose=1,
        )

    def fit(self, data: np.ndarray):
        self.kmeans.fit(data)

    def encode(self, motion: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(motion).astype(np.int32)

    def save(self, path: str | Path):
        joblib.dump(self.kmeans, path)

    def load(self, path: str | Path):
        self.kmeans = joblib.load(path)


def load_motion_tokenizer(tokenizer_path: str | Path, normalizer_path: str | Path):
    motion_tok = MotionTokenizer()
    motion_tok.load(tokenizer_path)
    norm = Normalizer()
    norm.load(normalizer_path)
    return motion_tok, norm


def encode_motion_from_dir(motion_dirname: str | Path, motion_tokenizer: MotionTokenizer, norm: Normalizer) -> np.ndarray:
    motion = load_smplx_sequence(motion_dirname, include_betas=False)
    motion = preprocess_motion(motion)
    motion = norm.transform(motion)
    return motion_tokenizer.encode(motion)


# =========================
# 3) Audio tokenization with EnCodec
# =========================

def tokenize_audio_encodec(audio_path: str | Path, bandwidth: float = 6.0) -> np.ndarray:
    """
    Returns:
        codes: [n_q, T]
    """
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)

    # BULLETPROOF FIX: Use soundfile directly instead of torchaudio
    wav_np, sr = sf.read(str(audio_path), dtype="float32")
    
    # Soundfile loads as (frames, channels), PyTorch expects (channels, frames)
    wav = torch.from_numpy(wav_np).t() 
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Convert audio to Encodec's expected sample rate and channels
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        encoded_frames = model.encode(wav)

    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)  # [B, n_q, T]
    return codes.squeeze(0).cpu().numpy().astype(np.int32)


def audio_tokens_to_text(codes: np.ndarray) -> str:
    n_q, T = codes.shape
    parts = []
    for t in range(T):
        for q in range(n_q):
            parts.append(f"<a_{q}_{int(codes[q, t])}>")
    return " ".join(parts)


def motion_tokens_to_text(tokens: np.ndarray) -> str:
    return " ".join(f"<m_{int(t)}>" for t in tokens.reshape(-1))


def build_text_row(audio_tokens: str, motion_tokens: str) -> str:
    return f"<|audio|> {audio_tokens} <|motion|> {motion_tokens}"


# =========================
# 4) Build training JSONL
# =========================

def build_joint_jsonl(
    csv_path: str | Path,
    motion_tokenizer_path: str | Path,
    normalizer_path: str | Path,
    output_jsonl: str | Path,
    audio_bandwidth: float = 6.0,
):
    df = pd.read_csv(csv_path)
    motion_tok, norm = load_motion_tokenizer(motion_tokenizer_path, normalizer_path)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            audio_path = row["audio_filename"]
            motion_dir = row["motion_dirname"]

            try:
                audio_codes = tokenize_audio_encodec(audio_path, bandwidth=audio_bandwidth)
                motion_codes = encode_motion_from_dir(motion_dir, motion_tok, norm)

                sample = {
                    "id": str(i),
                    "audio_filename": audio_path,
                    "motion_dirname": motion_dir,
                    "prompt": audio_tokens_to_text(audio_codes),
                    "completion": motion_tokens_to_text(motion_codes),
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping row {i}: {e}")


# =========================
# 5) Training token prep
# =========================

def _find_subsequence(seq: List[int], subseq: List[int]) -> int:
    if not subseq or len(subseq) > len(seq):
        return -1
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            return i
    return -1


def add_discrete_tokens(tokenizer, audio_codebook_size=1024, audio_num_codebooks=8, motion_vocab_size=1024):
    special = ["<|audio|>", "<|motion|>"]
    special += [f"<a_{q}_{i}>" for q in range(audio_num_codebooks) for i in range(audio_codebook_size)]
    special += [f"<m_{i}>" for i in range(motion_vocab_size)]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_for_training(
    example,
    tokenizer,
    max_seq_length=8192,
    prompt_max_length=2048,
    completion_max_length=6144,
):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]

    completion_ids = tokenizer(
        example["completion"],
        add_special_tokens=False,
        truncation=True,
        max_length=completion_max_length,
    )["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    active = sum(x != -100 for x in labels)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "active_label_tokens": active,
    }


# =========================
# 6) Debugging utilities
# =========================

def debug_example(example, tokenizer, max_seq_length=8192, prompt_max_length=2048, completion_max_length=6144):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]

    completion_ids = tokenizer(
        example["completion"],
        add_special_tokens=False,
        truncation=True,
        max_length=completion_max_length,
    )["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    active = sum(x != -100 for x in labels)
    print("prompt tokens:", len(prompt_ids))
    print("completion tokens:", len(completion_ids))
    print("total tokens:", len(input_ids))
    print("active label tokens:", active)
    return active

# def debug_example(example, tokenizer, max_length=4096):
#     # Call your actual training encoder so we see exactly what the model sees
#     enc = encode_for_training(example, tokenizer, max_length=max_length)
    
#     input_ids = enc["input_ids"]
#     labels = enc["labels"]
    
#     # Count how many tokens are NOT masked out by -100
#     active = sum(x != -100 for x in labels)
    
#     print(f"seq len: {len(input_ids)} | active label tokens: {active}")
#     return active


# =========================
# 6) Fine-tuning with Unsloth
# =========================

def finetune(
    base_model_name: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    max_seq_length: int = 8192,
    prompt_max_length: int = 2048,
    completion_max_length: int = 6144,
    load_in_4bit: bool = True,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Adjust these if your tokenizer sizes differ.
    tokenizer = add_discrete_tokens(tokenizer)

    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")
    for i in range(min(3, len(dataset))):
        debug_example(dataset[i], tokenizer, max_seq_length=max_seq_length, prompt_max_length=prompt_max_length, completion_max_length=completion_max_length)
    dataset = dataset.map(lambda ex: encode_for_training(ex, tokenizer, max_seq_length=max_seq_length, prompt_max_length=prompt_max_length, completion_max_length=completion_max_length), num_proc=2)

    print(dataset[0]["active_label_tokens"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        max_steps=100,
        logging_steps=5,
        save_steps=50,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(str(Path(output_dir) / "lora"))
    tokenizer.save_pretrained(str(Path(output_dir) / "lora"))


# =========================
# 7) Example usage
# =========================
if __name__ == "__main__":
    # 1) Build joint JSONL:
    # build_joint_jsonl(
    #     csv_path="dataset_mapping.csv",
    #     motion_tokenizer_path="motion_tokenizer_artifacts/tokenizer.pkl",
    #     normalizer_path="motion_tokenizer_artifacts/normalizer.npz",
    #     output_jsonl="speech_motion_train.jsonl",
    #     audio_bandwidth=6.0,
    # )

    # 2) Finetune:
    finetune(
        base_model_name="unsloth/orpheus-3b-0.1-pretrained",
        train_jsonl="speech_motion_train.jsonl",
        output_dir="speech_motion_outputs",
        max_seq_length=8192,
        prompt_max_length=2048,
        completion_max_length=6144,
        load_in_4bit=True,
    )
    pass