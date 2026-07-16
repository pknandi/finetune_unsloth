from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import re
import torch
import joblib
import numpy as np
import soundfile as sf

from pathlib import Path

from unsloth import FastLanguageModel
from peft import PeftModel

from encodec import EncodecModel
from encodec.utils import convert_audio

# =========================
# TOKENIZER FIX
# MUST MATCH TRAINING EXACTLY
# =========================
def add_discrete_tokens(
    tokenizer,
    audio_codebook_size=1024,
    audio_num_codebooks=8,
    motion_vocab_size=1024,
):
    special = ["<|audio|>", "<|motion|>"]

    special += [
        f"<a_{q}_{i}>"
        for q in range(audio_num_codebooks)
        for i in range(audio_codebook_size)
    ]

    special += [f"<m_{i}>" for i in range(motion_vocab_size)]

    tokenizer.add_special_tokens(
        {"additional_special_tokens": special}
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =========================
# 1) AUDIO ENCODING
# =========================
def tokenize_audio_encodec(
    audio_path: str | Path,
    bandwidth: float = 6.0,
    max_duration_sec: int = 10,
) -> np.ndarray:

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)

    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        frames_to_read = int(sr * max_duration_sec)

        wav_np = f.read(
            frames=frames_to_read,
            dtype="float32",
        )

    wav = torch.from_numpy(wav_np).t()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    wav = convert_audio(
        wav,
        sr,
        model.sample_rate,
        model.channels,
    )

    wav = wav.unsqueeze(0)

    with torch.no_grad():
        encoded_frames = model.encode(wav)

    codes = torch.cat(
        [frame[0] for frame in encoded_frames],
        dim=-1,
    )

    return codes.squeeze(0).cpu().numpy().astype(np.int32)


def audio_tokens_to_text(codes: np.ndarray) -> str:
    n_q, T = codes.shape
    parts = []
    for t in range(T):
        for q in range(n_q):
            parts.append(f"<a_{q}_{int(codes[q, t])}>")
    return "".join(parts)


# =========================
# 2) MOTION DECODING
# =========================
def decode_motion_tokens(
    motion_ids: list[int],
    tokenizer_path: str,
    normalizer_path: str,
) -> np.ndarray:

    # Reverse KMeans lookup
    kmeans = joblib.load(tokenizer_path)
    centroids = kmeans.cluster_centers_

    motion_norm = centroids[motion_ids]

    # Reverse normalization
    norm_data = np.load(normalizer_path)

    mean = norm_data["mean"]
    std = norm_data["std"]

    motion_feat = (motion_norm * std) + mean

    # Recover translation from velocity
    transl_vel = motion_feat[:, -3:]

    transl = np.cumsum(transl_vel, axis=0)

    final_motion = np.concatenate(
        [motion_feat[:, :-3], transl],
        axis=-1,
    )

    return final_motion


# =========================
# 3) LOAD MODEL CORRECTLY
# =========================
def load_model_and_tokenizer(lora_model_dir: str, base_model: str):

    print("Loading base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=8192,
        load_in_4bit=True,
    )

    print("Adding custom motion/audio tokens...")

    tokenizer = add_discrete_tokens(tokenizer)

    model.resize_token_embeddings(len(tokenizer))

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        lora_model_dir,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


# =========================
# 4) MAIN INFERENCE
# =========================
def generate_motion_from_audio(
    audio_path: str,
    lora_model_dir: str,
    tokenizer_path: str,
    normalizer_path: str,
    output_npy_path: str,
    base_model: str,
):

    print("1. Extracting Audio Tokens...")
    audio_codes = tokenize_audio_encodec(audio_path)
    audio_text = audio_tokens_to_text(audio_codes)
    prompt = f"<|audio|>{audio_text}<|motion|>" # <-- Removed the spaces

    print("2. Loading LLM + LoRA...")

    model, tokenizer = load_model_and_tokenizer(
        lora_model_dir, base_model
    )

    print("3. Generating Motion Tokens...")

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7,
            top_k=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False,
    )

    completion_text = output_text.split("<|motion|>")[-1]

    motion_ids_str = re.findall(
        r"<m_(\d+)>",
        completion_text,
    )

    motion_ids = [int(x) for x in motion_ids_str]

    if len(motion_ids) == 0:
        print("Model failed to generate motion tokens.")
        return

    print(f"Generated {len(motion_ids)} motion tokens.")

    print("4. Decoding Motion Tokens...")

    final_smplx_array = decode_motion_tokens(
        motion_ids=motion_ids,
        tokenizer_path=tokenizer_path,
        normalizer_path=normalizer_path,
    )

    np.save(output_npy_path, final_smplx_array)

    print(f"Saved motion to: {output_npy_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run Speech-to-Motion Inference"
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lora_model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--normalizer_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_npy_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--base_model", 
        type=str, 
        default="unsloth/orpheus-3b-0.1-pretrained"
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.output_npy_path),
        exist_ok=True,
    )

    generate_motion_from_audio(
        audio_path=args.audio_path,
        lora_model_dir=args.lora_model_dir,
        tokenizer_path=args.tokenizer_path,
        normalizer_path=args.normalizer_path,
        output_npy_path=args.output_npy_path,
        base_model=args.base_model  
    )