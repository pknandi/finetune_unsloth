# speech_to_motion_pipeline.py

from __future__ import annotations
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wandb
import soundfile as sf

from unsloth import FastLanguageModel
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import TrainingArguments, Trainer, default_data_collator

# IMPORT THE NEW ARCHITECTURE
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion

# =========================
# 1) Audio tokenization with EnCodec
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
    return "".join(parts) # NO SPACES

def motion_tokens_to_text(tokens: np.ndarray) -> str:
    return "".join(f"<m_{int(t)}>" for t in tokens.reshape(-1)) # NO SPACES

# =========================
# 2) Build training JSONL
# =========================
def build_joint_jsonl(
    csv_path: str | Path,
    tokenizer_path: str | Path,
    normalizer_path: str | Path,
    output_jsonl: str | Path,
    audio_bandwidth: float = 6.0,
    max_duration_sec: float = 10.0,
):
    df = pd.read_csv(csv_path)
    
    motion_tok = VQVAETokenizer()
    motion_tok.load(tokenizer_path)
    norm = Normalizer()
    norm.load(normalizer_path)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    audio_fps = 75
    motion_fps = 15  # VQVAE 2x Compression (30 / 2)
    audio_to_motion_ratio = int(audio_fps // motion_fps)  # 5

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            try:
                audio_codes = tokenize_audio_encodec(row["audio_filename"], bandwidth=audio_bandwidth)

                motion = load_smplx_sequence(row["motion_dirname"])
                motion = preprocess_motion(motion)
                motion = norm.transform(motion)
                motion_codes = motion_tok.encode(motion)

                # Exact 5:1 Temporal Alignment Math
                actual_audio_sec = audio_codes.shape[1] / audio_fps
                actual_motion_sec = motion_codes.shape[0] / motion_fps
                valid_sec = min(actual_audio_sec, actual_motion_sec, max_duration_sec)

                raw_a_frames = int(valid_sec * audio_fps)
                max_a_frames = (raw_a_frames // audio_to_motion_ratio) * audio_to_motion_ratio
                max_m_frames = max_a_frames // audio_to_motion_ratio
                
                audio_codes = audio_codes[:, :max_a_frames]
                motion_codes = motion_codes[:max_m_frames]

                sample = {
                    "id": str(i),
                    "prompt": f"<|audio|>{audio_tokens_to_text(audio_codes)}<|motion|>",
                    "completion": motion_tokens_to_text(motion_codes),
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping row {i}: {e}")

# =========================
# 3) Training prep & Debug
# =========================
def add_discrete_tokens(tokenizer, audio_codebook_size=1024, audio_num_codebooks=8, motion_vocab_size=1024):
    special = ["<|audio|>", "<|motion|>"]
    special += [f"<a_{q}_{i}>" for q in range(audio_num_codebooks) for i in range(audio_codebook_size)]
    special += [f"<m_{i}>" for i in range(motion_vocab_size)]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode_for_training(example, tokenizer, max_seq_length=8192):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=True, truncation=True, max_length=7000)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False, truncation=True, max_length=1000)["input_ids"]

    if len(completion_ids) > 0 and completion_ids[-1] != tokenizer.eos_token_id:
        completion_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    active = sum(x != -100 for x in labels)
    return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids), "labels": labels, "active_label_tokens": active}

def debug_example(example, tokenizer, max_seq_length=8192):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=True, truncation=True, max_length=7000)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False, truncation=True, max_length=1000)["input_ids"]
    
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
# 4) Fine-tuning
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
    # RESTORED: Checkpoint resuming logic
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

    # RESTORED: WandB naming
    wandb.init(project="speech-to-motion", name="orpheus-3b-finetune")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    
    tokenizer = add_discrete_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    model = FastLanguageModel.get_peft_model(
        model, r=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64, lora_dropout=0.0, bias="none", use_gradient_checkpointing=True, random_state=3407,
        modules_to_save=["embed_tokens", "lm_head"]
    )

    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(train_jsonl), split="train")
    
    # RESTORED: Debug example printing
    for i in range(min(3, len(dataset))):
        debug_example(dataset[i], tokenizer, max_seq_length=max_seq_length)

    dataset = dataset.map(lambda ex: encode_for_training(ex, tokenizer, max_seq_length), num_proc=2)

    # RESTORED: logging_steps, run_name
    args = TrainingArguments(
        output_dir=str(output_dir), 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        learning_rate=2e-4, 
        warmup_steps=10, 
        max_steps=max_steps, 
        logging_steps=logging_steps, # Fixed
        save_steps=save_steps,
        bf16=torch.cuda.is_available(), 
        fp16=not torch.cuda.is_available(), 
        optim="adamw_torch",
        weight_decay=0.01, 
        lr_scheduler_type="cosine", 
        report_to="wandb",
        run_name="orpheus-3b-finetune" # Fixed
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=default_data_collator)
    
    # RESTORED: pass resume_ckpt to trainer
    trainer.train(resume_from_checkpoint=resume_ckpt) 

    model.save_pretrained(str(Path(output_dir) / "lora"))
    tokenizer.save_pretrained(str(Path(output_dir) / "lora"))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dataset", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--csv_path", type=str, default="datasets/training_dataset_mapping.csv")
    parser.add_argument("--tokenizer_path", type=str, default="motion_tokenizer_artifacts/tokenizer.pt")
    parser.add_argument("--normalizer_path", type=str, default="motion_tokenizer_artifacts/normalizer.npz")
    parser.add_argument("--output_jsonl", type=str, default="datasets/speech_motion_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="speech_motion_outputs")
    parser.add_argument("--base_model", type=str, default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--max_steps", type=int, default=2000)
    
    # RESTORED: logging arguments
    parser.add_argument("--logging_steps", type=int, default=5, help="Log metrics to W&B every X steps")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume from.")

    args = parser.parse_args()

    if args.build_dataset:
        print("--- Step 1: Building Joint JSONL Dataset ---")
        build_joint_jsonl(args.csv_path, args.tokenizer_path, args.normalizer_path, args.output_jsonl)
        print(f"Dataset successfully saved to {args.output_jsonl}\n")
        
    if args.train:
        print("--- Step 2: Starting Unsloth Fine-Tuning ---")
        finetune(
            base_model_name=args.base_model, 
            train_jsonl=args.output_jsonl, 
            output_dir=args.output_dir, 
            max_steps=args.max_steps, 
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            resume_from_checkpoint=args.resume_from_checkpoint
        )