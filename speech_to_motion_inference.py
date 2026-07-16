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
    
    max_a_frames = min(audio_codes.shape[1], 750)
    max_a_frames = (max_a_frames // 10) * 10
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
    
    print("3. Generating Motion Tokens (with Repetition Penalty)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=True,          # Turned ON to allow smooth motion
            temperature=0.2,         # Very low temp prevents random flailing
            repetition_penalty=1.1,  # CRITICAL: Prevents the model from freezing
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

    transl_vel = motion_feat[:, -3:]
    transl = np.cumsum(transl_vel, axis=0)
    final_smplx_array = np.concatenate([motion_feat[:, :-3], transl], axis=-1)

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