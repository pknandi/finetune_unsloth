# speech_to_motion_inference.py

from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import re
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from unsloth import FastLanguageModel
from peft import PeftModel
from encodec import EncodecModel
from encodec.utils import convert_audio

from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer

def add_discrete_tokens(tokenizer):
    special = ["<|audio|>", "<|motion|>"]
    special += [f"<a_{q}_{i}>" for q in range(8) for i in range(1024)]
    special += [f"<m_{i}>" for i in range(1024)]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_audio_encodec(audio_path: str, bandwidth: float = 6.0) -> np.ndarray:
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
    return "".join(f"<a_{q}_{int(codes[q, t])}>" for t in range(T) for q in range(n_q)) 

def generate_motion_from_audio(audio_path, lora_model_dir, tokenizer_path, normalizer_path, output_npy_path, base_model):
    print("1. Extracting and Aligning Audio Tokens...")
    audio_codes = tokenize_audio_encodec(audio_path)
    
    audio_to_motion_ratio = 5  # VQVAE 2x compression: 75 audio-fps / 15 motion-fps
    max_a_frames = min(audio_codes.shape[1], 750)
    max_a_frames = (max_a_frames // audio_to_motion_ratio) * audio_to_motion_ratio
    audio_codes = audio_codes[:, :max_a_frames]
    
    prompt = f"<|audio|>{audio_tokens_to_text(audio_codes)}<|motion|>"

    print("2. Loading Model Safely...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, 
        max_seq_length=8192, 
        load_in_4bit=True
    )
    tokenizer = add_discrete_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, lora_model_dir)
    
    if getattr(model.config, "tie_word_embeddings", False):
        model.base_model.model.lm_head.weight = model.base_model.model.embed_tokens.weight
        
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    print("3. Generating Motion Tokens (greedy, no repetition penalty)...")
    # Greedy decoding with repetition_penalty=1.0: this pipeline is currently being
    # trained on a tiny, intentionally-overfit dataset where the correct motion-token
    # sequence for a given clip legitimately repeats the same token 60-90% of the time
    # (long static holds). do_sample + repetition_penalty actively fought that learned
    # distribution, pushing the model off of what it actually memorized. Revisit once
    # training data is large/diverse enough that repeated tokens are no longer the norm.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_text = tokenizer.decode(outputs[0], skip_special_tokens=False).split("<|motion|>")[-1]
    motion_ids = [int(x) for x in re.findall(r"<m_(\d+)>", completion_text)]
    if not motion_ids:
        print("Model failed to generate motion tokens.")
        return

    # DEBUG PRINT: This will show you if the LLM is actually generating diverse motion!
    print(f"   -> Generated {len(motion_ids)} tokens.")
    print(f"   -> First 15 tokens: {motion_ids[:15]}")

    print("4. VQ-VAE Decoding...")
    motion_tok = VQVAETokenizer()
    motion_tok.load(tokenizer_path)
    norm = Normalizer()
    norm.load(normalizer_path)

    motion_feat = motion_tok.decode(np.array(motion_ids))
    motion_feat = (motion_feat * norm.std) + norm.mean

    # motion_feat layout matches preprocess_motion's output: [go_sin(3), go_cos(3),
    # body(63), left_hand(45), right_hand(45), transl(3)] = 162 dims.
    # Undo the sin/cos wrap-around-safe encoding to recover a plain axis-angle
    # global_orient. Translation is stored as absolute world position (already
    # un-normalized above) — no velocity integration, so no accumulating drift.
    global_orient = np.arctan2(motion_feat[:, 0:3], motion_feat[:, 3:6])
    body_and_hands = motion_feat[:, 6:159]
    transl = motion_feat[:, 159:162]

    # Root translation is physically low-frequency (a body can't oscillate its
    # pelvis several cm per frame), but codebook quantization noise on the transl
    # channels un-normalizes into exactly that kind of frame-level wobble, which
    # renders as the character floating/gliding. A short centered moving average
    # (~0.37s at 30fps output) removes the wobble while preserving real walking
    # trajectories, which live at much lower frequencies.
    win = 11
    if len(transl) >= win:
        kernel = np.ones(win) / win
        pad = win // 2
        padded = np.pad(transl, ((pad, pad), (0, 0)), mode="edge")
        transl = np.stack(
            [np.convolve(padded[:, c], kernel, mode="valid") for c in range(3)], axis=-1
        )

    final_smplx_array = np.concatenate([global_orient, body_and_hands, transl], axis=-1)

    np.save(output_npy_path, final_smplx_array)
    print(f"Saved motion to: {output_npy_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--lora_model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--normalizer_path", type=str, required=True)
    parser.add_argument("--output_npy_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="unsloth/llama-3-8b-bnb-4bit")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_npy_path), exist_ok=True)
    generate_motion_from_audio(**vars(args))