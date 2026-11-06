"""Signal probe: does the audio carry information about the motion, and which audio features carry
the most? Trains a small transformer (no LLM) on a run's 10 s windows to predict each 15 fps motion
frame's RVQ tokens from audio aligned to that frame (+-~1 s of context) and, with --history, the
previous motion frame's tokens (what the LLM sees). Evaluated on the run's val windows three ways:
  right  the window's own audio
  other  another val window's audio (row k + n//2): breaks speaker identity and timing
  shift  the window's own audio rolled by 5 s: keeps the speaker, breaks the timing
Gap = wrong NLL - right NLL, in nats per token. The LLM's audio-shuffle gap on the same val set was
+0.007 (scripts/audio_shuffle.py), so a clearly larger probe gap means the signal is in the data and
the LLM fails to use it.

usage: python scripts/signal_probe.py <run_dir> <cache_dir> --feat encodec8|encodec2|mel|hubert [--history]
Last line: PROBE feat= history= right= other= shift= gap_other= gap_shift= gap_l0_shift= win_shift= speech_gap_shift=
"""
import os, sys, json, re, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("cache_dir")
ap.add_argument("--feat", required=True, choices=["encodec8", "encodec2", "mel", "hubert", "hubertkm", "none"])
ap.add_argument("--km", type=int, default=500, help="k-means units for --feat hubertkm")
ap.add_argument("--history", action="store_true")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.cache_dir, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda"
Q, T_M, T_A = 8, 150, 750           # motion levels, motion frames (15 fps), EnCodec frames (75 fps)
SILENCE = 62                          # EnCodec level-0 code that fills silent frames (measured 10-23)


def load_split(split):
    """Token arrays from the run's jsonl, cached. motion (N,150,8), encodec (N,750,8)."""
    path = f"{args.cache_dir}/tokens_{split}.npz"
    if os.path.exists(path):
        d = np.load(path); return d["motion"], d["encodec"]
    mot, enc = [], []
    for line in open(f"{args.run_dir}/speech_motion_{split}.jsonl"):
        r = json.loads(line)
        m = np.array(re.findall(r"<m_\d+_(\d+)>", r["completion"]), dtype=np.int16).reshape(-1, Q)
        a = np.array(re.findall(r"<a_\d+_(\d+)>", r["prompt"]), dtype=np.int16).reshape(-1, 8)
        if len(m) < T_M or len(a) < T_A:
            m = np.pad(m, ((0, T_M - len(m)), (0, 0)), constant_values=-1)[:T_M]
            a = np.pad(a, ((0, T_A - len(a)), (0, 0)), constant_values=SILENCE)[:T_A]
        mot.append(m); enc.append(a)
    mot, enc = np.stack(mot), np.stack(enc)
    np.savez(path, motion=mot, encodec=enc)
    return mot, enc


def windows(split):
    import csv
    return [(r["audio_filename"], float(r["start_sec"]))
            for r in csv.DictReader(open(f"{args.run_dir}/speech_motion_{split}_audio_windows.csv"))]


def read_window(path, start, sr_out):
    import soundfile as sf, librosa
    sr = sf.info(path).samplerate
    wav, _ = sf.read(path, start=int(start * sr), frames=int(10 * sr), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != sr_out:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
    n = 10 * sr_out
    return np.pad(wav, (0, max(0, n - len(wav))))[:n]


def _mel_one(w):
    import librosa
    wav = read_window(w[0], w[1], 24000)
    m = librosa.feature.melspectrogram(y=wav, sr=24000, n_fft=1024, hop_length=320, n_mels=80)
    return np.log(m + 1e-6).T[:T_A].astype(np.float16)                      # (750, 80) at 75 fps


def mel_feats(split):
    path = f"{args.cache_dir}/mel_{split}.npy"
    if os.path.exists(path):
        return np.load(path)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(16) as ex:
        out = np.stack(list(ex.map(_mel_one, windows(split), chunksize=8)))
    np.save(path, out)
    return out


def hubert_feats(split, layer=12):
    path = f"{args.cache_dir}/hubert_l{layer}_{split}.npy"
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    from transformers import HubertModel
    from concurrent.futures import ThreadPoolExecutor
    model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(dev).eval().half()
    ws = windows(split)
    with ThreadPoolExecutor(8) as ex:
        wavs = list(ex.map(lambda w: read_window(w[0], w[1], 16000), ws))
    out = np.zeros((len(ws), 499, 1024), np.float16)                          # 50 fps
    with torch.no_grad():
        for i in range(0, len(ws), 16):
            x = torch.tensor(np.stack(wavs[i:i + 16]), device=dev)
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
            h = model(x.half(), output_hidden_states=True).hidden_states[layer]
            out[i:i + len(x)] = h[:, :499].float().cpu().numpy()
    np.save(path, out)
    return out


def hubert_units(split, k):
    """k-means units (Lloyd's on GPU, fit on 300k train frames) of the layer-12 HuBERT features."""
    path = f"{args.cache_dir}/hubert_km{k}_{split}.npy"
    if os.path.exists(path):
        return np.load(path)
    cpath = f"{args.cache_dir}/hubert_km{k}_centroids.npy"
    if not os.path.exists(cpath):
        tr = hubert_feats("train")
        g = np.random.default_rng(0)
        rows = g.choice(len(tr), 600, replace=False)
        x = torch.tensor(np.asarray(tr[np.sort(rows)]).reshape(-1, 1024), device=dev).float()
        x = x[torch.randperm(len(x), device=dev)[:300000]]
        c = x[torch.randperm(len(x), device=dev)[:k]].clone()
        for _ in range(40):
            a = torch.cdist(x, c).argmin(1)
            s_ = torch.zeros_like(c).index_add_(0, a, x); n_ = torch.bincount(a, minlength=k).float()
            c = torch.where(n_[:, None] > 0, s_ / n_.clamp(min=1)[:, None], c)
        np.save(cpath, c.cpu().numpy())
    c = torch.tensor(np.load(cpath), device=dev)
    f = hubert_feats(split)
    out = np.zeros(f.shape[:2] + (1,), np.int16)
    for i in range(0, len(f), 64):
        x = torch.tensor(np.asarray(f[i:i + 64]), device=dev).float()
        out[i:i + 64, :, 0] = torch.cdist(x, c[None].expand(len(x), -1, -1)).argmin(-1).cpu().numpy()
    np.save(path, out)
    return out


def audio_input(split, enc):
    if args.feat == "encodec8": return enc
    if args.feat == "encodec2": return enc[:, :, :2]
    if args.feat == "mel": return mel_feats(split)
    if args.feat == "hubert": return hubert_feats(split)
    if args.feat == "hubertkm": return hubert_units(split, args.km)
    return np.zeros((len(enc), T_A, 1), np.float16)


class Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        discrete = args.feat.startswith("encodec") or args.feat == "hubertkm"
        self.discrete = discrete
        if discrete:
            nq = {"encodec8": 8, "encodec2": 2, "hubertkm": 1}[args.feat]
            self.a_emb = nn.ModuleList([nn.Embedding(1024, d) for _ in range(nq)])
        else:
            self.a_in = nn.LazyLinear(d)
        # native-rate conv, pool to 150 frames, then +-4 frames (~0.27 s each side per layer) of context
        self.a_conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.a_ctx = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Dropout(0.1),
                                   nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Dropout(0.1),
                                   nn.Conv1d(d, d, 5, padding=2))
        self.m_emb = nn.ModuleList([nn.Embedding(1025, d) for _ in range(Q)])   # 1024 = "no previous frame"
        self.pos = nn.Parameter(torch.randn(T_M, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.heads = nn.Linear(d, Q * 1024)
        self.register_buffer("mask", torch.triu(torch.full((T_M, T_M), float("-inf")), 1))

    def forward(self, audio, motion):
        if self.discrete:
            a = sum(e(audio[..., q]) for q, e in enumerate(self.a_emb))
        else:
            a = self.a_in(audio)
        a = self.a_conv(a.transpose(1, 2))
        a = F.adaptive_avg_pool1d(a, T_M)
        x = self.a_ctx(a).transpose(1, 2) + self.pos
        if args.history:
            prev = torch.cat([torch.full_like(motion[:, :1], 1024), motion[:, :-1]], 1).clamp(min=0)
            x = x + sum(e(prev[..., q]) for q, e in enumerate(self.m_emb))
        x = self.tf(x, mask=self.mask if args.history else None)
        return self.heads(x).view(x.shape[0], T_M, Q, 1024)


def nll(model, audio, motion, bs=64):
    """Per-row mean NLL over all levels and over level 0. Rows: (N,), (N,)."""
    model.eval(); out, out0 = [], []
    with torch.no_grad():
        for i in range(0, len(motion), bs):
            a = torch.as_tensor(np.asarray(audio[i:i + bs]), device=dev)
            a = a.long() if model.discrete else a.float()
            m = torch.as_tensor(motion[i:i + bs], device=dev).long()
            logits = model(a, m)
            l = F.cross_entropy(logits.reshape(-1, 1024).float(), m.reshape(-1).clamp(min=0),
                                reduction="none").view(m.shape)
            valid = (m >= 0).float()
            out.append(((l * valid).sum((1, 2)) / valid.sum((1, 2))).cpu())
            out0.append(((l[..., 0] * valid[..., 0]).sum(1) / valid[..., 0].sum(1)).cpu())
    return torch.cat(out).numpy(), torch.cat(out0).numpy()


tr_m, tr_e = load_split("train")
va_m, va_e = load_split("val")
tr_a, va_a = audio_input("train", tr_e), audio_input("val", va_e)
if args.feat in ("mel", "hubert"):
    # per-dim standardization from train stats (continuous features)
    mu = np.asarray(tr_a[:512], np.float32).mean((0, 1)); sd = np.asarray(tr_a[:512], np.float32).std((0, 1)) + 1e-4
    tr_a = ((np.asarray(tr_a, np.float32) - mu) / sd).astype(np.float16)
    va_a = ((np.asarray(va_a, np.float32) - mu) / sd).astype(np.float16)
n = len(va_m)
va_other = va_a[[(k + n // 2) % n for k in range(n)]]
va_shift = np.roll(va_a, va_a.shape[1] // 2, axis=1)       # 5 s circular shift, same speaker

model = Probe(args.d).to(dev)
with torch.no_grad():
    a0 = torch.as_tensor(np.asarray(tr_a[:2]), device=dev)
    model(a0.long() if model.discrete else a0.float(), torch.as_tensor(tr_m[:2], device=dev).long())
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
best, best_state, bad = 1e9, None, 0
for ep in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(tr_m))
    for i in range(0, len(perm), 32):
        idx = np.sort(perm[i:i + 32])
        a = torch.as_tensor(np.asarray(tr_a[idx]), device=dev)
        a = a.long() if model.discrete else a.float()
        m = torch.as_tensor(tr_m[idx], device=dev).long()
        logits = model(a, m)
        loss = F.cross_entropy(logits.reshape(-1, 1024).float(), m.reshape(-1), ignore_index=-1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    v = nll(model, va_a, va_m)[0].mean()
    print(f"epoch {ep:3d} train {loss.item():.3f} val {v:.3f}", flush=True)
    if v < best - 1e-3:
        best, bad = v, 0
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 8: break
model.load_state_dict(best_state)

r, r0 = nll(model, va_a, va_m)
o, o0 = nll(model, va_other, va_m)
s, s0 = nll(model, va_shift, va_m)
speech = (va_e[:, :, 0] != SILENCE).mean(1)
hi = speech >= np.median(speech)
print(f"PROBE feat={args.feat} history={int(args.history)} right={r.mean():.3f} other={o.mean():.3f} "
      f"shift={s.mean():.3f} gap_other={o.mean() - r.mean():+.3f} gap_shift={s.mean() - r.mean():+.3f} "
      f"gap_l0_shift={s0.mean() - r0.mean():+.3f} win_shift={(s > r).sum()}/{n} "
      f"speech_gap_shift={(s - r)[hi].mean():+.3f}/{(s - r)[~hi].mean():+.3f}", flush=True)
