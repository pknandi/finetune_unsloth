# run_inference.py

from __future__ import annotations

import re
import torch
import joblib
import numpy as np
import soundfile as sf
from pathlib import Path

from unsloth import FastLanguageModel
from encodec import EncodecModel
from encodec.utils import convert_audio


# =========================
# 1) Audio Encoding
# =========================
def tokenize_audio_encodec(audio_path: str | Path, bandwidth: float = 6.0, max_duration_sec: int = 5) -> np.ndarray:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)

    # Read the audio, but only up to max_duration_sec
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        frames_to_read = int(sr * max_duration_sec)
        wav_np = f.read(frames=frames_to_read, dtype="float32")

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
    return " ".join(parts)


# =========================
# 2) Motion Decoding
# =========================
def decode_motion_tokens(motion_ids: list[int], tokenizer_path: str, normalizer_path: str) -> np.ndarray:

    # 1. Reverse the K-Means Lookup
    kmeans = joblib.load(tokenizer_path)
    centroids = kmeans.cluster_centers_

    # Grab the 159-dimensional arrays for each predicted ID
    motion_norm = centroids[motion_ids]

    # 2. Reverse the Normalization
    norm_data = np.load(normalizer_path)
    mean = norm_data["mean"]
    std = norm_data["std"]

    motion_feat = (motion_norm * std) + mean

    # 3. Reverse Velocity to Absolute Translation
    # motion_feat layout: [global_orient(3), body(63), left_hand(45), right_hand(45), transl_vel(3)]
    transl_vel = motion_feat[:, -3:]

    transl = np.zeros_like(transl_vel)
    # Cumulative sum integrates velocity back into spatial position
    transl = np.cumsum(transl_vel, axis=0)

    # Rebuild the final array with absolute translation
    final_motion = np.concatenate([motion_feat[:, :-3], transl], axis=-1)

    return final_motion


# =========================
# 3) Main Inference Pipeline
# =========================
def generate_motion_from_audio(audio_path: str, lora_model_dir: str, tokenizer_path: str, normalizer_path: str, output_npy_path: str):
    print("1. Extracting Audio Tokens...")
    audio_codes = tokenize_audio_encodec(audio_path)
    audio_text = audio_tokens_to_text(audio_codes)

    # Format the exact prompt the model expects
    prompt = f"<|audio|> {audio_text} <|motion|>"

    print("2. Loading LLM (Unsloth)...")
    # Load the base model with your trained LoRA adapters merged on top
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_model_dir,
        max_seq_length=8192,
        load_in_4bit=True,
    )

    # Unsloth optimization for 2x faster inference
    FastLanguageModel.for_inference(model)

    print("3. Generating Motion Tokens...")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate tokens. max_new_tokens determines how long the motion will be.
    # 1500 tokens is roughly ~50 frames of motion depending on your frame rate.
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.7,  # Lower temperature = less creative, more accurate to training
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the output back to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Slice off the prompt so we only look at the model's generated text
    completion_text = output_text.split("<|motion|>")[-1]

    # Find all motion tokens e.g., <m_42> and extract the integer 42
    motion_ids_str = re.findall(r"<m_(\d+)>", completion_text)
    motion_ids = [int(x) for x in motion_ids_str]

    if not motion_ids:
        print("Model failed to generate any <m_X> tokens.")
        return

    print(f"Generated {len(motion_ids)} frames of motion.")

    print("4. Decoding Tokens to 3D Arrays...")
    final_smplx_array = decode_motion_tokens(motion_ids=motion_ids, tokenizer_path=tokenizer_path, normalizer_path=normalizer_path)

    np.save(output_npy_path, final_smplx_array)
    print(f"Success! Saved raw motion array to: {output_npy_path}")


if __name__ == "__main__":
    audio_file_name = "DAYLIFE_Doing_chores_together--186171-190520.wav"
    generate_motion_from_audio(
        audio_path=f"inference_data/input/{audio_file_name}",
        lora_model_dir="speech_motion_outputs/lora",
        tokenizer_path="motion_tokenizer_artifacts/tokenizer.pkl",
        normalizer_path="motion_tokenizer_artifacts/normalizer.npz",
        output_npy_path=f"inference_data/output/generated_motion_{audio_file_name.split('.')[0]}.npy",
    )
