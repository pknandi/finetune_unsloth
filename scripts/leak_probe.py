"""How far ahead in the audio does a model really listen? (paper option B: hidden future-audio leakage)
For a few eval windows and cut times tau, the audio after tau is replaced by noise; features are rebuilt the way the
model is used (full: one HuBERT pass over the window, as build_dense.py; stream: HubertFeaturizer.stream), and the
model's teacher-forced logits on the real motion tokens are compared. The first frame t whose logits move heard
audio after tau; frame t's prediction is due at the end of its slot, (t + 1) / 15 s, so the measured lookahead is
tau - (t + 1) / 15 (max over windows and cuts; <= 0 means online-honest).

usage: python scripts/leak_probe.py <ckpt_dir> <full|stream> [--data outputs/run-full-dense-rel-oct28] [--n 5]
Prints: LEAK ckpt=<> feats=<> lookahead_s=<max> per-cut=<...>
"""
import os, sys, csv, json, random, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
from speech_to_motion_aligned import read_window, HubertFeaturizer, T_M, FPS_M
from speech_to_motion_small import load_trained

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("feats", choices=["full", "stream"])
ap.add_argument("--data", default="outputs/run-full-dense-rel-oct28"); ap.add_argument("--n", type=int, default=5)
args = ap.parse_args()
L = json.load(open(f"{args.data}/meta.json"))["lookahead"]
DUR, S = (T_M + L) / FPS_M, T_M + L
va_w = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))
va_m = np.load(f"{args.data}/val_motion.npy")
NF = np.load(f"{args.data}/val_hubert.npy", mmap_mode="r").shape[1]
random.seed(0); idx = sorted(random.sample(range(len(va_w)), 60))[:args.n]
model = load_trained(args.ckpt)[0]
feat = HubertFeaturizer()


def features(w):
    if args.feats == "stream":
        return feat.stream(w, NF, S)
    h = feat(w[None])[0, :2 * NF].astype(np.float32)
    return ((h[0::2] + h[1::2]) / 2).astype(np.float16)


@torch.no_grad()
def logits(f, codes):
    return model(torch.as_tensor(f, device="cuda")[None], torch.as_tensor(codes, device="cuda").long()[None]).float()[0]


worst, per = -1e9, []
for i in idx:
    w = read_window(va_w[i]["audio_filename"], float(va_w[i]["start_sec"]), DUR)
    codes = va_m[i].astype(np.int64)
    base = logits(features(w), codes)
    for tau in (3.0, 5.0, 7.0):
        w2 = w.copy(); c = int(tau * 16000)
        w2[c:] = np.random.default_rng(0).normal(0, w.std() + 1e-3, len(w) - c)
        d = (logits(features(w2), codes) - base).abs().amax((1, 2)).cpu().numpy()       # per frame
        moved = np.where(d > 1e-3)[0]
        la = tau - (moved[0] + 1) / FPS_M if len(moved) else -tau
        per.append(round(float(la), 3)); worst = max(worst, la)
print(f"LEAK ckpt={args.ckpt} feats={args.feats} lookahead_s={worst:.3f} per-cut={per}", flush=True)
