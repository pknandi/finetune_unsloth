# speech_to_motion_pipeline.py

from __future__ import annotations

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

import argparse
import json
from pathlib import Path
from typing import List, Optional

# Unsloth imported FIRST to apply optimizations
from unsloth import FastLanguageModel

import soundfile as sf
import joblib
import numpy as np
import pandas as pd
import torch
import wandb

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
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)

    wav_np, sr = sf.read(str(audio_path), dtype="float32")
    wav = torch.from_numpy(wav_np).t()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        encoded_frames = model.encode(wav)

    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
    return codes.squeeze(0).cpu().numpy().astype(np.int32)

def audio_tokens_to_text(codes: np.ndarray) -> str:
    n_q, T = codes.shape
    parts = []
    for t in range(T):
        for q in range(n_q):
            parts.append(f"<a_{q}_{int(codes[q, t])}>")
    return "".join(parts)

def motion_tokens_to_text(tokens: np.ndarray) -> str:
    return "".join(f"<m_{int(t)}>" for t in tokens.reshape(-1))

# =========================
# 4) Build training JSONL
# =========================

def build_joint_jsonl(
    csv_path: str | Path,
    motion_tokenizer_path: str | Path,
    normalizer_path: str | Path,
    output_jsonl: str | Path,
    audio_bandwidth: float = 6.0,
    max_duration_sec: float = 10.0,  # Added safe duration limit
):
    df = pd.read_csv(csv_path)
    motion_tok, norm = load_motion_tokenizer(motion_tokenizer_path, normalizer_path)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    audio_fps = 75
    motion_fps = 30

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            audio_path = row["audio_filename"]
            motion_dir = row["motion_dirname"]

            try:
                audio_codes = tokenize_audio_encodec(audio_path, bandwidth=audio_bandwidth)
                motion_codes = encode_motion_from_dir(motion_dir, motion_tok, norm)

                # FIX 1: Exact Temporal Alignment & Clipping
                actual_audio_sec = audio_codes.shape[1] / audio_fps
                actual_motion_sec = motion_codes.shape[0] / motion_fps
                
                # Take the strict minimum to prevent any dangling inputs/outputs
                valid_sec = min(actual_audio_sec, actual_motion_sec, max_duration_sec)
                
                max_a_frames = int(valid_sec * audio_fps)
                max_m_frames = int(valid_sec * motion_fps)
                
                audio_codes = audio_codes[:, :max_a_frames]
                motion_codes = motion_codes[:max_m_frames]

                # FIX 2: Correct Modality Markers
                sample = {
                    "id": str(i),
                    "audio_filename": audio_path,
                    "motion_dirname": motion_dir,
                    "prompt": f"<|audio|>{audio_tokens_to_text(audio_codes)}<|motion|>",
                    "completion": f"{motion_tokens_to_text(motion_codes)}",
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping row {i}: {e}")

# =========================
# 5) Training token prep
# =========================

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
):
    # Truncate prompt leaving enough room for completion
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=6144, 
    )["input_ids"]

    completion_ids = tokenizer(
        example["completion"],
        add_special_tokens=False,
        truncation=True,
        max_length=2048,
    )["input_ids"]

    # FIX 3: Manually append the EOS token so the model learns when to stop
    if len(completion_ids) > 0 and completion_ids[-1] != tokenizer.eos_token_id:
        completion_ids.append(tokenizer.eos_token_id)

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

def debug_example(
    example,
    tokenizer,
    max_seq_length=8192,
):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=6144,
    )["input_ids"]

    completion_ids = tokenizer(
        example["completion"],
        add_special_tokens=False,
        truncation=True,
        max_length=2048,
    )["input_ids"]
    
    if len(completion_ids) > 0 and completion_ids[-1] != tokenizer.eos_token_id:
        completion_ids.append(tokenizer.eos_token_id)

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

# =========================
# 7) Fine-tuning with Unsloth
# =========================

def finetune(
    base_model_name: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    max_steps: int = 2000,
    logging_steps: int = 5,
    save_steps: int = 1000,
    resume_from_checkpoint: str | Path | None = None,
):
    resume_ckpt = None
    if resume_from_checkpoint is not None:
        resume_ckpt = str(resume_from_checkpoint)
        print(f"Resuming from specified checkpoint: {resume_ckpt}")
    else:
        ckpt_dirs = sorted(
            [d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if ckpt_dirs:
            resume_ckpt = str(ckpt_dirs[-1])
            print(f"Auto-resuming from latest checkpoint: {resume_ckpt}")
        else:
            print("No checkpoints found — starting fresh.")

    wandb.init(project="speech-to-motion", name="orpheus-3b-finetune")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    tokenizer = add_discrete_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        modules_to_save=["embed_tokens", "lm_head"],
    )

    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")
    
    for i in range(min(3, len(dataset))):
        debug_example(dataset[i], tokenizer, max_seq_length=max_seq_length)
        
    dataset = dataset.map(
        lambda ex: encode_for_training(ex, tokenizer, max_seq_length=max_seq_length),
        num_proc=2,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=10,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name="orpheus-3b-finetune",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    model.save_pretrained(str(Path(output_dir) / "lora"))
    tokenizer.save_pretrained(str(Path(output_dir) / "lora"))
    wandb.finish()

# =========================
# 8) Example usage
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech to Motion Fine-Tuning Pipeline")

    parser.add_argument("--build_dataset", action="store_true", help="Process audio/motion into a JSONL dataset")
    parser.add_argument("--train", action="store_true", help="Run the Unsloth fine-tuning process")
    parser.add_argument("--csv_path", type=str, default="datasets/training_dataset_mapping.csv")
    parser.add_argument("--tokenizer_path", type=str, default="motion_tokenizer_artifacts/tokenizer.pkl")
    parser.add_argument("--normalizer_path", type=str, default="motion_tokenizer_artifacts/normalizer.npz")
    parser.add_argument("--output_jsonl", type=str, default="datasets/speech_motion_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="speech_motion_outputs")
    parser.add_argument("--base_model", type=str, default="unsloth/orpheus-3b-0.1-pretrained")
    parser.add_argument("--max_steps", type=int, default=2000, help="Total number of training steps")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log metrics to W&B every X steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save a model checkpoint every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from.")

    args = parser.parse_args()

    if not args.build_dataset and not args.train:
        print("Please specify a step to run using --build_dataset or --train (or both).")
        parser.print_help()
        exit()

    if args.build_dataset:
        print("--- Step 1: Building Joint JSONL Dataset ---")
        build_joint_jsonl(
            csv_path=args.csv_path,
            motion_tokenizer_path=args.tokenizer_path,
            normalizer_path=args.normalizer_path,
            output_jsonl=args.output_jsonl,
            audio_bandwidth=6.0,
            max_duration_sec=10.0, # Added the hard 10-second cap
        )
        print(f"Dataset successfully saved to {args.output_jsonl}\n")

    if args.train:
        print("--- Step 2: Starting Unsloth Fine-Tuning ---")
        finetune(
            base_model_name=args.base_model,
            train_jsonl=args.output_jsonl,
            output_dir=args.output_dir,
            max_seq_length=8192,
            load_in_4bit=True,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print("Fine-tuning complete!")