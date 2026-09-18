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

MOTION_NUM_CODEBOOKS = 2  # must match MotionVQVAE num_quantizers and training-side add_discrete_tokens

def add_discrete_tokens(tokenizer):
    special = ["<|audio|>", "<|motion|>"]
    special += [f"<a_{q}_{i}>" for q in range(8) for i in range(1024)]
    special += [f"<m_{q}_{i}>" for q in range(MOTION_NUM_CODEBOOKS) for i in range(1024)]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_audio_encodec(audio_path: str, bandwidth: float = 6.0) -> np.ndarray:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    # RVQ codebook lookups are nearest-neighbor on continuous encoder output, so tiny
    # CPU-vs-GPU floating-point differences can flip a token index right at a decision
    # boundary (measured ~0.7% of positions on a real clip — not a bug, just numeric
    # non-determinism between devices). build_joint_jsonl encodes training audio on GPU
    # when available; doing the same here keeps inference-time tokens on the same
    # footing as what the model was actually trained on, instead of a silent train/
    # inference device skew on top of whatever the model already has to generalize over.
    if torch.cuda.is_available():
        model = model.cuda()

    wav_np, sr = sf.read(str(audio_path), dtype="float32")
    wav = torch.from_numpy(wav_np).t()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        encoded_frames = model.encode(wav)
        
    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
    return codes.squeeze(0).cpu().numpy().astype(np.int32)

def audio_tokens_to_text(codes: np.ndarray) -> str:
    n_q, T = codes.shape
    return "".join(f"<a_{q}_{int(codes[q, t])}>" for t in range(T) for q in range(n_q)) 

def generate_motion_from_audio(audio_path, lora_model_dir, tokenizer_path, normalizer_path, output_npy_path, base_model,
                                do_sample=True, temperature=1.0, top_p=0.9, repetition_penalty=1.2,
                                no_repeat_ngram_size=0):
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
    
    print(f"3. Generating Motion Tokens (do_sample={do_sample}, temperature={temperature}, "
          f"top_p={top_p}, repetition_penalty={repetition_penalty})...")
    # Was greedy + repetition_penalty=1.0 while this pipeline trained on a tiny,
    # intentionally-overfit dataset (same clip in train and test) — there, the "correct"
    # motion-token sequence legitimately repeated the same token 60-90% of the time (long
    # static holds), and do_sample/repetition_penalty actively fought that learned
    # distribution, pushing the model off what it had memorized. Now that training data is
    # the full, diverse Embody-3D corpus, greedy decoding is the wrong default: it always
    # takes the single most likely next token, which collapses onto exactly the kind of
    # repeated-token degenerate loop repetition_penalty exists to prevent, and produces one
    # fixed output per audio input with no ability to sample multiple candidate gestures.
    # Sampling with a repetition penalty is the standard choice once outputs are meant to
    # reflect real learned diversity rather than a memorized answer. Pass do_sample=False
    # to go back to the old greedy/memorization-check behavior for a specific debug run.
    #
    # temperature=0.7 (the first full-dataset default) was a real bug, not just "too
    # weak": dividing logits by a value BELOW 1.0 SHARPENS the distribution — it makes
    # the model MORE deterministic/repetitive, the opposite of what fights a repeat
    # collapse. Measured on a real full-dataset generation: 265 of 299 frame-to-frame
    # body-pose deltas were identical to 5 decimal places (a single repeated 4-token RVQ
    # block replayed for 8.8 of 10 seconds) — repetition_penalty=1.15 was not remotely
    # enough to break a lock that confident.
    #
    # temperature=1.0 (neutral — the raw trained distribution, neither sharpened nor
    # flattened) + repetition_penalty=1.2 is chosen from a real head-to-head, not just
    # "more must be better": (temp, rep) = (1.2, 1.3), (1.0, 1.15), (1.0, 1.2), (0.9, 1.2)
    # were all tested against the same checkpoint/audio. All four broke the repeat
    # collapse (longest constant-delta run: 1 frame, vs. 265). None clearly beat the
    # collapsed run's 764mm MPJPE, and all showed elevated jerk (15-32x GT, vs. the
    # collapsed run's misleadingly-low 3x — a frozen signal has near-zero jerk almost by
    # definition, so that comparison was never fair). (1.0, 1.2) was the least-bad of the
    # four (699.9mm MPJPE) — this is the best decoding config found, not evidence that
    # decoding alone fixes generation quality. It doesn't: every anti-collapse setting
    # traded "frozen" for "noisy," which means the model's own learned distribution
    # beyond its single dominant mode isn't coherent yet — a training-data/capacity
    # problem (see run.sh's MAX_STEPS comment), not something a decoding flag can paper
    # over. no_repeat_ngram_size is available but defaults off (0): it HARD-blocks any
    # repeated n-gram, which also blocks legitimately-static motion (holding still while
    # listening) — a blunter tool than temperature/repetition_penalty, and untested
    # against the noise/collapse tradeoff above.
    #
    # IMPORTANT — these settings do not transparently transfer across checkpoints, and
    # re-tuning them on a new one is not optional. Re-verified on run-full-aug30 (10000
    # steps, ~2.5x the training of the checkpoint the numbers above came from, on a
    # different motion-token vocabulary — the VQ-VAE was retrained too): (1.0, 1.2) still
    # avoids collapse (1-frame run) but jerk got WORSE, not better, at 52x GT (vs. 20x
    # before) — loss curve was healthy and still decreasing through the last epoch, so
    # this isn't undertraining regressing, it's that a decoding config tuned for one
    # model's probability landscape doesn't carry over to another's. A second sweep
    # against THIS checkpoint found no config that both avoids collapse and controls
    # jerk: --greedy freezes 277/299 frames (jerk 8.9x); (0.7, 1.15) freezes 153/299
    # (jerk 9.0x, better MPJPE-vs-PCK than (1.0,1.2) despite the partial freeze); (1.0,
    # 1.2) is the only zero-collapse setting found, at the cost of high jerk. Kept as the
    # default because collapse was the more severe, visually-confusing failure to begin
    # with, but this is a real, unresolved tradeoff, not a solved one — re-sweep again
    # against any future checkpoint rather than assuming these numbers hold.
    generate_kwargs = dict(
        max_new_tokens=1500,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        # Only meaningful (and only accepted without warnings) under sampling.
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    completion_text = tokenizer.decode(outputs[0], skip_special_tokens=False).split("<|motion|>")[-1]
    # Multi-level RVQ tokens: <m_q_i> where q is the residual level. Each temporal
    # block is a run of MOTION_NUM_CODEBOOKS tokens in level order (q=0,1,2,3).
    pairs = [(int(q), int(i)) for q, i in re.findall(r"<m_(\d+)_(\d+)>", completion_text)]
    if not pairs:
        print("Model failed to generate motion tokens.")
        return

    # Group into complete blocks, tolerating malformed output: keep only runs
    # where the levels appear in the exact expected order, drop anything partial.
    Q = MOTION_NUM_CODEBOOKS
    blocks = []
    idx = 0
    while idx + Q <= len(pairs):
        chunk = pairs[idx:idx + Q]
        if [q for q, _ in chunk] == list(range(Q)):
            blocks.append([i for _, i in chunk])
            idx += Q
        else:
            idx += 1  # resync: skip one token and look for the next full block
    if not blocks:
        print("Model generated motion tokens but no complete level-ordered blocks.")
        return
    motion_ids = np.array(blocks, dtype=np.int64)  # (T, Q)

    # DEBUG PRINT: This will show you if the LLM is actually generating diverse motion!
    print(f"   -> Generated {len(pairs)} tokens -> {motion_ids.shape[0]} temporal blocks.")
    print(f"   -> First 3 blocks: {motion_ids[:3].tolist()}")

    print("4. VQ-VAE Decoding...")
    motion_tok = VQVAETokenizer()
    motion_tok.load(tokenizer_path)
    norm = Normalizer()
    norm.load(normalizer_path)

    motion_feat = motion_tok.decode(motion_ids)
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
    # renders as the character floating/gliding. A centered moving average
    # (~0.5s at 30fps output) removes the wobble while preserving real walking
    # trajectories, which live at much lower frequencies. Measured on real data:
    # widening from 11->15 improves jitter and GT-velocity correlation with no
    # added lag; lag starts appearing beyond ~19, so this is deliberately not
    # wider despite further (diminishing) correlation gains out there.
    win = 15
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
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling — the old behavior, "
                             "correct only for the tiny-overfit-set memorization check "
                             "(same session in train and test), not for the full corpus.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Below 1.0 SHARPENS the distribution (more repetitive, not less) — "
                             "keep this >= 1.0 unless you have a specific reason to go lower.")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="Hard-blocks any repeated n-gram (0 = off). Stronger than "
                             "repetition_penalty but also blocks legitimately-static motion "
                             "(e.g. holding still while listening) — try only if repetition "
                             "is still severe with the other flags.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_npy_path), exist_ok=True)
    generate_motion_from_audio(
        audio_path=args.audio_path,
        lora_model_dir=args.lora_model_dir,
        tokenizer_path=args.tokenizer_path,
        normalizer_path=args.normalizer_path,
        output_npy_path=args.output_npy_path,
        base_model=args.base_model,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )