# speech_to_motion_aligned.py
"""Time-aligned speech-to-motion LLM.

Why (measured 2026-10-23, docs/EXPERIMENT_LOG.md): the EnCodec-prompt LLM ignored the audio (shuffle
gap +0.007 nats/token on 2880 clips, = the 1000-row model's). A small probe that sees audio aligned to
each motion frame found almost no timing signal in EnCodec tokens (+0.009) or mel (+0.001), and a clear
one in continuous HuBERT features (+0.054, 83% of clips; scripts/signal_probe.py). k-means HuBERT units
kept only a third of it (+0.019). So the audio goes in as continuous features, not tokens.

Layout (one sequence per 10 s window, LOOKAHEAD = L motion frames of audio ahead of the motion, since
motion leads speech by ~0.1-0.2 s, scripts/sync_check.py):
    [BOS] [a_0 .. a_{L-1}]  then for t = 0..149:  [a_{t+L}] [m_t^0 .. m_t^7]
a_k = projector(HuBERT-large layer-12 features pooled to motion frame k, 15 fps); the window's audio
is extracted with L frames of extra tail so a_{t+L} is real audio. m_t^q = motion RVQ token (own
embedding table, not LLM vocab). The LLM's lm_head is replaced by an 8x1024 motion head; loss is the
level-q slice's cross-entropy at the positions that predict m_t^q (a_{t+L} -> m_t^0, m_t^q -> m_t^{q+1}).
The LLM vocabulary is untouched, so no full embed_tokens/lm_head copies (the old format's 26-32 GB).

Data: <data_dir>/{train,val}_motion.npy (N,150,Q) int16 (-1 = no frame), {train,val}_hubert.npy
(N,530,1024) fp16 memmap, built by `--build` from a run's jsonl (motion tokens) + windows csv.
"""
from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
import re, json, csv, math, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_M, FPS_M = 150, 15                 # motion frames per 10 s window, motion fps (tokenizer 2x downsample of 30)
HUBERT, HUBERT_LAYER, HUBERT_FPS = "facebook/hubert-large-ll60k", 12, 50
DEFAULT_LOOKAHEAD = 8                # motion frames (0.53 s)
BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"


# ---------------------------------------------------------------- audio features
def read_window(path, start_sec, dur_sec, sr_out=16000):
    import soundfile as sf, librosa
    sr = sf.info(path).samplerate
    wav, _ = sf.read(path, start=int(start_sec * sr), frames=int(dur_sec * sr), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != sr_out:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
    n = int(dur_sec * sr_out)
    return np.pad(wav, (0, max(0, n - len(wav))))[:n]      # zero-pad past the end of the file


class HubertFeaturizer:
    def __init__(self, device="cuda"):
        from transformers import HubertModel
        self.model = HubertModel.from_pretrained(HUBERT).to(device).eval().half()
        self.device = device

    @torch.no_grad()
    def __call__(self, wavs: np.ndarray) -> np.ndarray:
        """(B, samples) 16 kHz -> (B, frames, 1024) fp16, frames = samples // 320 - 1."""
        x = torch.tensor(wavs, device=self.device)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
        h = self.model(x.half(), output_hidden_states=True).hidden_states[HUBERT_LAYER]
        return h.float().cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def stream(self, wav: np.ndarray, nf: int, slots: int, bs: int = int(os.environ.get("STREAM_BS", 32))) -> np.ndarray:
        """Causal 25 fps features (nf, 1024) for one window, as an online system would have them: HuBERT is
        bidirectional, so __call__'s frame t depends on audio after t. Here the 25 fps frames pooled into
        motion slot j (SmallS2M.encode_audio: adaptive pool nf -> slots) come from a HuBERT pass over the
        audio up to the end of slot j ((j + 1) / 15 s), the rest replaced by silence; each frame is kept
        from the first slot that uses it."""
        out = np.zeros((nf, 1024), np.float32)
        done = 0
        jobs = []                                                   # (cut in samples, first frame, end frame)
        for j in range(slots):
            e = min(nf, -(-(j + 1) * nf // slots))
            if e > done:
                jobs.append((min(len(wav), round((j + 1) * 16000 / FPS_M)), done, e)); done = e
        for b in range(0, len(jobs), bs):
            n = max(jobs[min(b + bs, len(jobs)) - 1][0], 400 + 320 * jobs[min(b + bs, len(jobs)) - 1][2] * 2)
            x = np.zeros((len(jobs[b:b + bs]), min(len(wav), n)), np.float32)   # silence only up to the batch's need
            for k, (cut, _, _) in enumerate(jobs[b:b + bs]):
                w = wav[:cut]; x[k, :cut] = (w - w.mean()) / (w.std() + 1e-5)
            x = torch.tensor(x, device=self.device).half()
            h = self.model(x, output_hidden_states=True).hidden_states[HUBERT_LAYER].float()
            h = h[:, :min(2 * nf, h.shape[1] // 2 * 2)]
            h = ((h[:, 0::2] + h[:, 1::2]) / 2).cpu().numpy()
            for k, (_, s, e) in enumerate(jobs[b:b + bs]):
                out[s:e] = h[k, s:e]
        return out.astype(np.float16)


def n_hubert_frames(lookahead):
    return int((T_M + lookahead) / FPS_M * HUBERT_FPS) - 1


def build_data(run_dir, data_dir, lookahead=DEFAULT_LOOKAHEAD, num_quantizers=8):
    """Motion tokens from the run's jsonl completions, HuBERT features from its windows csv (same row
    order: build_joint_jsonl writes both). Audio spans the window plus `lookahead` motion frames."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(data_dir, exist_ok=True)
    feat = HubertFeaturizer()
    dur = (T_M + lookahead) / FPS_M
    nf = n_hubert_frames(lookahead)
    for split in ["train", "val"]:
        rows = [json.loads(l) for l in open(f"{run_dir}/speech_motion_{split}.jsonl")]
        mot = np.full((len(rows), T_M, num_quantizers), -1, np.int16)
        for i, r in enumerate(rows):
            m = np.array(re.findall(r"<m_\d+_(\d+)>", r["completion"]), dtype=np.int16).reshape(-1, num_quantizers)
            mot[i, :len(m)] = m[:T_M]
        np.save(f"{data_dir}/{split}_motion.npy", mot)
        wins = [(w["audio_filename"], float(w["start_sec"]))
                for w in csv.DictReader(open(f"{run_dir}/speech_motion_{split}_audio_windows.csv"))]
        assert len(wins) == len(rows), f"{split}: {len(wins)} windows vs {len(rows)} jsonl rows"
        out = np.lib.format.open_memmap(f"{data_dir}/{split}_hubert.npy", "w+", np.float16, (len(rows), nf, 1024))
        with ThreadPoolExecutor(8) as ex:
            for i in range(0, len(wins), 16):
                wavs = list(ex.map(lambda w: read_window(w[0], w[1], dur), wins[i:i + 16]))
                out[i:i + len(wavs)] = feat(np.stack(wavs))[:, :nf]
                if i % 320 == 0:
                    print(f"{split} hubert {i}/{len(wins)}", flush=True)
        out.flush()
        with open(f"{data_dir}/{split}_windows.csv", "w", newline="") as f:
            csv.writer(f).writerows([("audio_filename", "start_sec")] + wins)
    json.dump({"lookahead": lookahead, "num_quantizers": num_quantizers, "run_dir": str(run_dir)},
              open(f"{data_dir}/meta.json", "w"))
    print(f"built {data_dir}")


# ---------------------------------------------------------------- model
class MotionIO(nn.Module):
    """Everything outside the LLM: audio projector, motion-token embedding, motion head."""
    def __init__(self, hidden, num_quantizers, lookahead, feat_mean=None, feat_std=None):
        super().__init__()
        self.Q, self.L = num_quantizers, lookahead
        self.register_buffer("feat_mean", torch.zeros(1024) if feat_mean is None else torch.as_tensor(feat_mean).float())
        self.register_buffer("feat_std", torch.ones(1024) if feat_std is None else torch.as_tensor(feat_std).float())
        self.a_conv = nn.Sequential(nn.Conv1d(1024, 1024, 5, padding=2), nn.GELU())
        self.a_out = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.a_type = nn.Parameter(torch.zeros(hidden))
        self.m_emb = nn.Embedding(num_quantizers * 1024, hidden)
        self.m_mask = nn.Parameter(torch.zeros(hidden))     # stands in for a dropped motion frame (--hist_drop)
        self.head = nn.Linear(hidden, num_quantizers * 1024, bias=False)
        nn.init.normal_(self.m_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def audio_slots(self, feats):
        """(B, F, 1024) HuBERT -> (B, T_M + L, hidden), one slot per motion frame."""
        x = (feats.float() - self.feat_mean) / self.feat_std
        x = self.a_conv(x.transpose(1, 2))
        x = F.adaptive_avg_pool1d(x, T_M + self.L).transpose(1, 2)
        return self.a_out(x) + self.a_type

    def motion_embed(self, codes):
        """(B, T, Q) codes (-1 -> 0) -> (B, T, Q, hidden)."""
        q = torch.arange(self.Q, device=codes.device)
        return self.m_emb(codes.clamp(min=0).long() + q * 1024)


def layout(Q, L):
    """Positions in the sequence: audio slot k is at aslot[k]; motion (t, q) at mpos[t, q]."""
    n = 1 + L + T_M * (1 + Q)
    aslot = np.zeros(T_M + L, np.int64)
    mpos = np.zeros((T_M, Q), np.int64)
    aslot[:L] = 1 + np.arange(L)
    for t in range(T_M):
        base = 1 + L + t * (1 + Q)
        aslot[t + L] = base
        mpos[t] = base + 1 + np.arange(Q)
    # the position whose output predicts (t, q): the token right before it
    pred = mpos - 1
    return n, aslot, mpos, pred


def build_inputs(lm, io, feats, codes, bos_emb, hist_drop=0.0):
    """Full teacher-forced input embeddings (B, n, hidden). hist_drop: fraction of motion frames whose
    input embeddings (all levels) are replaced by io.m_mask, so the model can't lean on history alone."""
    B = codes.shape[0]
    n, aslot, mpos, _ = layout(io.Q, io.L)
    slots = io.audio_slots(feats)
    memb = io.motion_embed(codes)
    if hist_drop > 0:
        drop = torch.rand(B, codes.shape[1], 1, 1, device=codes.device) < hist_drop
        memb = torch.where(drop, io.m_mask.to(memb.dtype), memb)
    x = torch.zeros(B, n, slots.shape[-1], device=codes.device, dtype=slots.dtype)
    x[:, 0] = bos_emb
    x[:, torch.as_tensor(aslot, device=codes.device)] = slots
    x[:, torch.as_tensor(mpos.reshape(-1), device=codes.device)] = memb.reshape(B, -1, memb.shape[-1]).to(x.dtype)
    return x


def motion_nll(lm, io, feats, codes, bos_emb, hist_drop=0.0):
    """Per-token NLL (B, T_M, Q) and validity mask."""
    _, _, _, pred = layout(io.Q, io.L)
    x = build_inputs(lm, io, feats, codes, bos_emb, hist_drop).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = lm(inputs_embeds=x, use_cache=False).logits
    lg = logits[:, torch.as_tensor(pred.reshape(-1), device=codes.device)].float()
    lg = lg.view(codes.shape[0], T_M, io.Q, io.Q, 1024)
    lg = lg[:, :, torch.arange(io.Q), torch.arange(io.Q)]          # level q's slice at (t, q)
    nll = F.cross_entropy(lg.reshape(-1, 1024), codes.clamp(min=0).reshape(-1).long(), reduction="none")
    return nll.view(codes.shape), (codes >= 0)


def load_lm(lora_dir=None, r=32, lora_dropout=0.0):
    from unsloth import FastLanguageModel
    lm, tok = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=2048, load_in_4bit=True)
    if lora_dir is None:
        lm = FastLanguageModel.get_peft_model(
            lm, r=r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=2 * r, lora_dropout=lora_dropout, bias="none", use_gradient_checkpointing=True, random_state=3407)
    else:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, lora_dir, is_trainable=False)
    bos = lm.get_input_embeddings().weight[tok.bos_token_id].detach().float().clone()
    return lm, tok, bos


class _EmbedOverride:
    """unsloth's KV-cache path (past_key_values given) only takes input_ids, one token per call, and
    embeds them itself. This hook swaps that embedding for ours: set .x, call with a dummy id."""
    def __init__(self, lm):
        self.x = None
        lm.get_input_embeddings().register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        return out if self.x is None else self.x.to(out.dtype).view(out.shape)


def cached_step(lm, e, past):
    """Feed one embedding (hidden,) after `past`; returns (last logits, new past)."""
    ov = lm._embed_override
    ov.x = e
    try:
        pos = torch.tensor([[past[0][0].shape[-2]]], device="cuda")
        out = lm(input_ids=torch.zeros(1, 1, dtype=torch.long, device="cuda"), past_key_values=past,
                 position_ids=pos, use_cache=True, return_dict=True)
    finally:
        ov.x = None
    return out.logits[0, -1].float(), out.past_key_values


def attach_head(lm, io):
    """Replace the LLM's lm_head with the motion head (shares io.head's weight)."""
    lm.base_model.model.lm_head = io.head
    if not hasattr(lm, "_embed_override"):
        lm._embed_override = _EmbedOverride(lm)


def save(out_dir, lm, io, extra):
    os.makedirs(out_dir, exist_ok=True)
    lm.save_pretrained(out_dir)
    torch.save({"io": io.state_dict(), "Q": io.Q, "L": io.L, **extra}, f"{out_dir}/motion_io.pt")


def load_trained(ckpt_dir):
    lm, tok, bos = load_lm(ckpt_dir)
    st = torch.load(f"{ckpt_dir}/motion_io.pt", map_location="cpu")
    io = MotionIO(lm.config.hidden_size, st["Q"], st["L"]).cuda()
    io.load_state_dict(st["io"], strict=False)       # checkpoints before --hist_drop have no m_mask
    attach_head(lm, io)
    lm.eval(); io.eval()
    return lm, io, bos.cuda()


# ---------------------------------------------------------------- train
def evaluate(lm, io, bos, feats, codes, idx, bs=4):
    """Mean NLL with the right audio, audio rolled 5 s (same speaker, wrong timing), another row's
    audio; level-0 versions; fraction of rows where the shifted audio is worse."""
    lm.eval(); io.eval()
    res = {k: [] for k in ["right", "shift", "other", "right0", "shift0", "other0"]}
    n = len(idx)
    with torch.no_grad():
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            ob = [idx[(k + n // 2) % n] for k in range(i, min(i + bs, n))]
            c = torch.as_tensor(codes[b], device="cuda")
            f = torch.as_tensor(np.asarray(feats[b]), device="cuda")
            variants = {"right": f, "shift": torch.roll(f, f.shape[1] // 2, 1),
                        "other": torch.as_tensor(np.asarray(feats[ob]), device="cuda")}
            for k, fv in variants.items():
                nll, m = motion_nll(lm, io, fv, c, bos)
                m = m.float()
                res[k] += ((nll * m).sum((1, 2)) / m.sum((1, 2))).tolist()
                res[k + "0"] += ((nll[..., 0] * m[..., 0]).sum(1) / m[..., 0].sum(1)).tolist()
    lm.train(); io.train()
    r = {k: float(np.mean(v)) for k, v in res.items()}
    r["win_shift"] = float(np.mean(np.array(res["shift"]) > np.array(res["right"])))
    return r


def train(args):
    torch.manual_seed(0); np.random.seed(0)
    meta = json.load(open(f"{args.data_dir}/meta.json"))
    Q, L = meta["num_quantizers"], meta["lookahead"]
    tr_m = np.load(f"{args.data_dir}/train_motion.npy"); va_m = np.load(f"{args.data_dir}/val_motion.npy")
    tr_f = np.load(f"{args.data_dir}/train_hubert.npy", mmap_mode="r")
    va_f = np.load(f"{args.data_dir}/val_hubert.npy", mmap_mode="r")
    if args.max_rows:
        tr_m, tr_f = tr_m[:args.max_rows], tr_f[:args.max_rows]
    samp = np.asarray(tr_f[np.linspace(0, len(tr_f) - 1, min(256, len(tr_f))).astype(int)], np.float32).reshape(-1, 1024)
    lm, tok, bos = load_lm(r=args.lora_r, lora_dropout=args.lora_dropout)
    io = MotionIO(lm.config.hidden_size, Q, L, samp.mean(0), samp.std(0) + 1e-4).cuda()
    attach_head(lm, io)
    bos = bos.cuda()
    lm.train(); io.train()
    lora = [p for p in lm.parameters() if p.requires_grad and p is not io.head.weight]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr},
                             {"params": list(io.parameters()), "lr": args.io_lr}], weight_decay=0.01)
    steps = args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    va_idx = list(np.random.RandomState(0).permutation(len(va_m))[:args.val_rows])
    tr_idx_eval = list(np.random.RandomState(1).permutation(len(tr_m))[:min(64, len(tr_m))])
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(f"{args.out_dir}/train_log.jsonl", "a")
    best, bad, step, t0 = 1e9, 0, 0, time.time()
    order = np.array([], int)
    while step < steps:
        tot = 0.0
        for _ in range(args.accum):
            if len(order) < args.batch:
                order = np.concatenate([order, np.random.permutation(len(tr_m))])
            b, order = np.sort(order[:args.batch]), order[args.batch:]
            c = torch.as_tensor(tr_m[b], device="cuda")
            f = torch.as_tensor(np.asarray(tr_f[b]), device="cuda")
            nll, m = motion_nll(lm, io, f, c, bos, args.hist_drop)
            loss = (nll * m).sum() / m.sum()
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(lora + list(io.parameters()), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            print(f"step {step} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            log.write(json.dumps({"step": step, "loss": tot}) + "\n"); log.flush()
        if step % args.eval_every == 0 or step == steps:
            ev = evaluate(lm, io, bos, va_f, va_m, va_idx)
            tr = evaluate(lm, io, bos, tr_f, tr_m, tr_idx_eval)
            print(f"EVAL step={step} right={ev['right']:.3f} shift={ev['shift']:.3f} other={ev['other']:.3f} "
                  f"gap_shift={ev['shift'] - ev['right']:+.3f} gap_other={ev['other'] - ev['right']:+.3f} "
                  f"gap0_shift={ev['shift0'] - ev['right0']:+.3f} win_shift={ev['win_shift']:.2f} "
                  f"train_right={tr['right']:.3f} train_gap_shift={tr['shift'] - tr['right']:+.3f}", flush=True)
            log.write(json.dumps({"step": step, "val": ev, "train": tr}) + "\n"); log.flush()
            save(f"{args.out_dir}/last", lm, io, {"step": step})
            if ev["right"] < best:
                best, bad = ev["right"], 0
                save(f"{args.out_dir}/best", lm, io, {"step": step})
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print(f"early stop: val NLL has not improved for {bad} evals (best {best:.3f})", flush=True)
                    break
    print("done", flush=True)


# ---------------------------------------------------------------- generate
@torch.no_grad()
def generate(lm, io, bos, feats, do_sample=False, temperature=1.0, top_p=0.9, sample_levels=None):
    """feats: (F, 1024) HuBERT for the window + lookahead. Returns (T_M, Q) ids."""
    Q, L = io.Q, io.L
    slots = io.audio_slots(torch.as_tensor(np.asarray(feats), device="cuda")[None])[0]

    def pick(logits, q):
        lg = logits.view(Q, 1024)[q]
        if not do_sample or (sample_levels is not None and q >= sample_levels):
            return int(lg.argmax())
        p = torch.softmax(lg / temperature, -1)
        sp, si = p.sort(descending=True)
        sp = sp * (sp.cumsum(0) - sp < top_p)
        return int(si[torch.multinomial(sp / sp.sum(), 1)])

    x = torch.cat([bos[None].to(slots.dtype), slots[:L + 1]])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm(inputs_embeds=x[None].to(torch.bfloat16), use_cache=True)
    logits, past = out.logits[0, -1].float(), out.past_key_values
    ids = np.zeros((T_M, Q), np.int64)
    for t in range(T_M):
        for q in range(Q):
            ids[t, q] = pick(logits, q)
            if t == T_M - 1 and q == Q - 1:
                break
            logits, past = cached_step(lm, io.m_emb.weight[q * 1024 + ids[t, q]], past)
        if t < T_M - 1:
            logits, past = cached_step(lm, slots[t + 1 + L], past)
    return ids


@torch.no_grad()
def check_kv(lm, io, bos, feats, codes, frames=3):
    """Teacher-forced logits vs the incremental (KV-cache) path used by generate(), on the first frames."""
    lm.eval(); io.eval()
    f = torch.as_tensor(np.asarray(feats), device="cuda")[None]
    c = torch.as_tensor(codes, device="cuda")[None]
    _, _, _, pred = layout(io.Q, io.L)
    x = build_inputs(lm, io, f, c, bos).to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = lm(inputs_embeds=x, use_cache=False).logits[0].float()
        out = lm(inputs_embeds=x[:, :pred[0, 0] + 1], use_cache=True)
    diffs, past = [(out.logits[0, -1].float() - full[pred[0, 0]]).abs().max().item()], out.past_key_values
    pos = pred[0, 0] + 1
    while pos <= pred[frames - 1, io.Q - 1]:
        lg, past = cached_step(lm, x[0, pos], past)
        diffs.append((lg - full[pos]).abs().max().item())
        pos += 1
    print(f"KV check: max |incremental - full| logit diff {max(diffs):.3f} over {len(diffs)} positions "
          f"(logit scale {full.abs().max().item():.1f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build <data_dir> from <run_dir>")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--check_kv", help="checkpoint dir: run check_kv on val row 0")
    ap.add_argument("--run_dir"); ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir")
    ap.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    ap.add_argument("--num_quantizers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--io_lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--hist_drop", type=float, default=0.0, help="fraction of motion frames masked in training")
    ap.add_argument("--patience", type=int, default=0, help="stop after N evals without val improvement (0 = off)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--val_rows", type=int, default=200)
    ap.add_argument("--max_rows", type=int, default=0, help="train on the first N rows only (0 = all)")
    args = ap.parse_args()
    if args.build:
        build_data(args.run_dir, args.data_dir, args.lookahead, args.num_quantizers)
    if args.train:
        train(args)
    if args.check_kv:
        lm, io, bos = load_trained(args.check_kv)
        check_kv(lm, io, bos, np.load(f"{args.data_dir}/val_hubert.npy", mmap_mode="r")[0],
                 np.load(f"{args.data_dir}/val_motion.npy")[0])
        t0 = time.time()
        ids = generate(lm, io, bos, np.load(f"{args.data_dir}/val_hubert.npy", mmap_mode="r")[0])
        print(f"generate: {ids.shape} in {time.time() - t0:.1f}s, {len(np.unique(ids[:, 0]))} distinct level-0 codes", flush=True)
