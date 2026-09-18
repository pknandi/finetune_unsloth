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
from tqdm import tqdm

from unsloth import FastLanguageModel
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import TrainingArguments, Trainer, default_data_collator

# IMPORT THE NEW ARCHITECTURE
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion

# =========================
# 1) Audio tokenization with EnCodec
# =========================
def tokenize_audio_encodec(audio_path: str | Path, bandwidth: float = 6.0, model: EncodecModel = None,
                            max_duration_sec: float | None = None) -> np.ndarray:
    # model is optional only for backward compatibility (e.g. one-off scripts calling
    # this directly). build_joint_jsonl below always passes a shared instance — building
    # a fresh EncodecModel per call is fine for a handful of rows but reconstructs the
    # whole model from scratch on every row, which at ~9,000+ rows (full Embody-3D corpus)
    # turns into hours of pure model-reload overhead before any real encoding happens.
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)

    wav_np, sr = sf.read(str(audio_path), dtype="float32")
    if max_duration_sec is not None:
        # Embody-3D clips run 35s-300s+ (measured); build_joint_jsonl only ever keeps the
        # first max_duration_sec of tokens anyway (see valid_sec below), so encoding the
        # full file was pure waste — one 300s clip took as long as EnCodec-encoding thirty
        # 10s ones. Slicing axis 0 (samples) works whether wav_np is mono (samples,) or
        # multi-channel (samples, channels) per soundfile's layout. A no-op for anything
        # already shorter than max_duration_sec.
        wav_np = wav_np[: int(sr * max_duration_sec)]
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
    parts = []
    for t in range(T):
        for q in range(n_q):
            parts.append(f"<a_{q}_{int(codes[q, t])}>")
    return "".join(parts) # NO SPACES

def motion_tokens_to_text(tokens: np.ndarray) -> str:
    # tokens: (T, Q) — Q residual-VQ levels per temporal block, mirroring the
    # audio format <a_q_i>. Interleaved per block: <m_0_x><m_1_y><m_2_z><m_3_w>...
    T, Q = tokens.shape
    return "".join(f"<m_{q}_{int(tokens[t, q])}>" for t in range(T) for q in range(Q)) # NO SPACES

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

    # Built once and reused for every row — see the comment on tokenize_audio_encodec.
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(audio_bandwidth)
    # Measured ~120x faster per clip on GPU than CPU (9.9s -> 0.08s warm, on a 10s clip) —
    # the dominant cost at full-dataset scale. tokenize_audio_encodec moves the waveform
    # to whatever device this model is on, so this one line is what actually engages it.
    if torch.cuda.is_available():
        encodec_model = encodec_model.cuda()

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    audio_fps = 75
    motion_fps = 15  # VQVAE 2x Compression (30 / 2)
    audio_to_motion_ratio = int(audio_fps // motion_fps)  # 5

    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Building joint JSONL"):
            try:
                audio_codes = tokenize_audio_encodec(row["audio_filename"], model=encodec_model,
                                                      max_duration_sec=max_duration_sec)

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
def add_discrete_tokens(tokenizer, audio_codebook_size=1024, audio_num_codebooks=8,
                        motion_vocab_size=1024, motion_num_codebooks=2):
    # motion_num_codebooks must match MotionVQVAE's num_quantizers (vqvae_motion_tokenizer.py)
    special = ["<|audio|>", "<|motion|>"]
    special += [f"<a_{q}_{i}>" for q in range(audio_num_codebooks) for i in range(audio_codebook_size)]
    special += [f"<m_{q}_{i}>" for q in range(motion_num_codebooks) for i in range(motion_vocab_size)]
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
# 3b) Muon optimizer
# =========================
# Muon (Jordan et al.) orthogonalizes each 2D weight matrix's momentum-accumulated
# gradient before applying it, via a fast Newton-Schulz iteration approximating the
# matrix's polar factor (its SVD's U @ V^T) instead of computing an actual SVD.
# Empirically converges faster per step than AdamW on hidden linear-layer weights
# (popularized by the nanoGPT speedrun work). It is explicitly NOT a drop-in
# replacement everywhere: it's defined for 2D weight matrices only — the original
# design keeps embeddings, the unembedding/lm_head, and any 1D params (biases, norm
# weights) on AdamW. In this LoRA setup that split falls out naturally: LoRA's A/B
# matrices for every adapted linear layer (q/k/v/o_proj, gate/up/down_proj) are
# exactly the 2D hidden weights Muon targets; embed_tokens/lm_head are
# modules_to_save (fully retrained, not LoRA-adapted) and are embedding/unembedding
# matrices, so they stay on AdamW. IMPORTANT: this speeds up reaching a given
# training loss — it is an optimizer, not a regularizer, and does not by itself
# prevent overfitting past the point that's actually good for held-out data (see
# the eval_dataset / load_best_model_at_end wiring in finetune() below, which is
# what actually catches that).
def _zeropower_via_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2, f"Muon's orthogonalization step expects a 2D matrix, got shape {tuple(G.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315  # quintic coefficients from the published iteration
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """A single optimizer that dispatches each param group to Muon or AdamW by a
    per-group `use_muon` flag, set when building the param groups in finetune(). One
    optimizer object (not two) is what Trainer(optimizers=...) needs."""

    def __init__(self, param_groups):
        for g in param_groups:
            g.setdefault("use_muon", False)
            if g["use_muon"]:
                g.setdefault("lr", 0.02)          # Muon's natural LR scale is much larger than Adam's —
                g.setdefault("momentum", 0.95)    # the orthogonalized update has unit-ish singular values,
                g.setdefault("weight_decay", 0.0) # not raw-gradient scale, so it isn't comparable to Adam's lr.
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("lr", 2e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._step_muon if group["use_muon"] else self._step_adam)(group)
        return loss

    def _step_muon(self, group):
        lr, momentum, wd, ns_steps = group["lr"], group["momentum"], group["weight_decay"], group["ns_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            # A spiked or already-clipped-through-Inf gradient (Inf * clip_coef==0 -> NaN)
            # must not touch the momentum buffer or the weights — once NaN gets into either,
            # it propagates forever and silently kills the run (seen in run-full-sep8: a single
            # Inf grad_norm at step 119 left every step after 133 training a dead model). Skip
            # this param for this step instead; momentum buffer stays at its last good value.
            if not torch.isfinite(g).all():
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            g = g.add(buf, alpha=momentum)  # Nesterov-style lookahead
            g = _zeropower_via_newton_schulz(g, steps=ns_steps)
            if wd != 0:
                p.data.mul_(1 - lr * wd)
            # Aspect-ratio-normalized step size, so the same lr is sane whether the
            # matrix is wide (LoRA A: r x in_features) or tall (LoRA B: out_features x r).
            scale = 0.2 * max(g.shape[-2:]) ** 0.5
            p.data.add_(g, alpha=-lr * scale)

    def _step_adam(self, group):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all():
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(b1).add_(g, alpha=1 - b1)
            exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
            bias_c1 = 1 - b1 ** state["step"]
            bias_c2 = 1 - b2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / bias_c2 ** 0.5).add_(eps)
            if wd != 0:
                p.data.mul_(1 - lr * wd)
            p.data.addcdiv_(exp_avg, denom, value=-lr / bias_c1)


def build_muon_optimizer(model, muon_lr: float = 0.02, adam_lr: float = 2e-4) -> MuonWithAuxAdam:
    """Splits the model's trainable parameters into LoRA's 2D matrices (Muon) and
    everything else — embed_tokens/lm_head, mainly (AdamW). See the module-level
    comment above for why the split falls exactly on lora_A/lora_B by name."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("lora_A" in name or "lora_B" in name) and p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)
    print(f"Muon optimizer: {len(muon_params)} LoRA weight matrices on Muon, "
          f"{len(adam_params)} other trainable tensors (embed_tokens/lm_head, etc.) on AdamW.")
    return MuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr),
        dict(params=adam_params, use_muon=False, lr=adam_lr),
    ])


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
    run_name: str = "speech-to-motion",
    val_jsonl: str | Path | None = None,
    use_muon: bool = True,
    muon_lr: float = 0.02,
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

    # run_name identifies this run.sh RUN_NAME (e.g. "run-full-sep3") in the wandb
    # dashboard — was hardcoded to "orpheus-3b-finetune" (a stale placeholder unrelated
    # to anything in this project: no "Orpheus" elsewhere in the codebase, and the base
    # model is 8B, not 3B). Each finetune() call opens a fresh wandb run even when
    # resuming from a checkpoint (no id= passed to wandb.init to continue the prior
    # run), so the step count is appended to tell separate attempts under the same
    # RUN_NAME apart in the run list.
    wandb_run_name = f"{run_name}-step{max_steps}"
    wandb.init(project="speech-to-motion", name=wandb_run_name)

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

    # Held-out validation: previously nothing here at all — training tracked train/loss
    # only, so it had zero visibility into generalization. Measured consequence on a real
    # run (run-full-sep3, 10000 steps): teacher-forced accuracy on a genuinely held-out
    # clip dropped from 62%/58% (RVQ levels 0/1) at step 3000 to 48%/48% at step 10000,
    # while train/loss kept improving the whole time — classic overfitting, invisible
    # until a manual post-hoc check. val_jsonl (built from dataset_to_csv.py's own
    # TEST_FRAC split, already existed, was just never fed to Trainer) fixes that:
    # eval_dataset + load_best_model_at_end restores whichever checkpoint actually
    # minimized held-out loss, not just whatever the last one happened to be, and
    # save_total_limit above no longer risks pruning it (Trainer keeps the best
    # checkpoint alive regardless of the limit once load_best_model_at_end is set).
    val_dataset = None
    if val_jsonl is not None:
        val_dataset = load_dataset("json", data_files=str(val_jsonl), split="train")
        val_dataset = val_dataset.map(lambda ex: encode_for_training(ex, tokenizer, max_seq_length), num_proc=2)
        print(f"Validation set: {len(val_dataset)} held-out examples from {val_jsonl}")

    # RESTORED: logging_steps, run_name
    #
    # Real crash on the first real run of this: HF's default eval loop keeps and
    # upcasts the FULL (batch, seq_len, vocab_size) logits tensor to fp32 every eval
    # batch, for a possible compute_metrics() to consume. This project's vocab is
    # inflated by the audio/motion special tokens (~139K entries), and
    # per_device_eval_batch_size silently defaults to 8 — 8x the batch size training
    # ever actually ran at (per_device_train_batch_size=1 below). Together: one eval
    # batch tried to materialize+upcast a single 26GB tensor with only 13GB free after
    # the model/optimizer state (torch.OutOfMemoryError: "Tried to allocate 26.02
    # GiB"). Fixed with prediction_loss_only=True (we only ever read eval_loss for
    # metric_for_best_model — no compute_metrics needs the logits, so this makes the
    # eval loop return just the scalar loss and skip materializing predictions
    # entirely) and per_device_eval_batch_size matching training's batch of 1.
    eval_kwargs = dict(
        eval_strategy="steps",
        eval_steps=save_steps,   # must align with save_steps for load_best_model_at_end
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        per_device_eval_batch_size=1,
    ) if val_dataset is not None else {}

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        # 10 was fine for AdamW-only runs (peak lr 2e-4), but Muon's group peaks at 0.02 —
        # 100x higher — and ramping there in 10 steps caused a real gradient explosion
        # (grad_norm hit Infinity at step 119 in run-full-sep8, which NaN-poisoned every
        # step after it once clipping divided by that Infinity). 200 steps gives the
        # orthogonalized Muon update room to stabilize before hitting full LR.
        warmup_steps=200,
        max_steps=max_steps,
        logging_steps=logging_steps, # Fixed
        save_steps=save_steps,
        # Each checkpoint saves full embed_tokens + lm_head (modules_to_save below) plus
        # their optimizer state, not just the LoRA delta — measured at ~12GB each. Without
        # a limit, a long run keeps every checkpoint indefinitely and can fill the disk
        # regardless of how much space you started with; 3 is enough to resume/compare
        # recent checkpoints without unbounded growth. Safe to combine with
        # load_best_model_at_end above (Trainer exempts the best checkpoint from pruning).
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        # optim= is only the fallback when use_muon=False below — Trainer skips building
        # its own optimizer entirely once one is passed via optimizers=, so this string
        # (and learning_rate= above) go unused whenever Muon is active.
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name=wandb_run_name,
        **eval_kwargs,
    )

    # See the module-level comment on MuonWithAuxAdam for why this optimizer choice
    # doesn't replace the eval_dataset fix above: Muon converges faster per step, it
    # doesn't know or care whether "faster" is still on the right side of overfitting —
    # only the held-out eval_loss above catches that, regardless of which optimizer got
    # you there.
    optimizer = build_muon_optimizer(model, muon_lr=muon_lr, adam_lr=2e-4) if use_muon else None
    trainer_kwargs = dict(model=model, args=args, train_dataset=dataset, data_collator=default_data_collator)
    if val_dataset is not None:
        trainer_kwargs["eval_dataset"] = val_dataset
    if optimizer is not None:
        trainer_kwargs["optimizers"] = (optimizer, None)  # None -> Trainer builds the cosine scheduler around it
    trainer = Trainer(**trainer_kwargs)

    # RESTORED: pass resume_ckpt to trainer
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # With load_best_model_at_end=True, trainer.train() already reloaded the best-eval-loss
    # checkpoint's weights into `model` before returning — this saves that one, not
    # whatever the final step happened to leave in memory (when val_dataset is None, this
    # is unchanged from before: just the last step's weights).
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
    parser.add_argument("--run_name", type=str, default="speech-to-motion",
                        help="Identifies this run in the wandb dashboard — pass run.sh's RUN_NAME "
                             "(e.g. run-full-sep3) so separate experiments are distinguishable there.")
    parser.add_argument("--val_output_jsonl", type=str, default=None,
                        help="Held-out validation JSONL (build with --build_dataset against the "
                             "*_test.csv split). Enables eval_dataset + load_best_model_at_end, "
                             "so the saved model is whichever checkpoint actually minimized "
                             "held-out loss, not just the last one. Skipped if not passed.")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                        help="muon: LoRA's 2D weight matrices on Muon, embed_tokens/lm_head on "
                             "AdamW (see MuonWithAuxAdam). adamw: the previous all-AdamW behavior.")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Muon's natural LR scale is much larger than Adam's (orthogonalized "
                             "updates, not raw-gradient scale) — not comparable to --learning_rate.")

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
            resume_from_checkpoint=args.resume_from_checkpoint,
            run_name=args.run_name,
            val_jsonl=args.val_output_jsonl,
            use_muon=(args.optimizer == "muon"),
            muon_lr=args.muon_lr,
        )