"""Continuous streaming (M4): minutes-long motion from a live audio stream with no real-motion seed and no resets.
The model works on 10.5 s windows; we roll a window forward by --hop seconds at a time. For each new window the
already emitted motion overlapping it is re-expressed in that window's frame (preprocess_motion: translation and,
for 6d_rel, heading relative to the window's first frame), encoded to tokens as the prefix, and the model generates
only the --hop new seconds from the streamed (causal) audio features of the window; the new frames are decoded
(causal decoder, one-sided smoothing), placed back in the world frame and appended, with the seam difference to the
last emitted frame faded out over the new frames only. Nothing already emitted changes.
The first window starts from no motion at all (unseeded).
Scores each 10 s segment of the stream against the real motion of the same segments (FD as
scripts/motion_dist_metrics.py, speed and jerk ratio to real), the pose jump at window boundaries, and the same for
a reset baseline (independent unseeded 10 s windows, concatenated).

usage: python scripts/continuous_stream.py <ckpt> <tok_dir> <out_dir> [--sigma -2.5] [--dur 60] [--hop 2] [--n 20]
       [--temp 1.3] [--guid 3.5] [--adapt 1.5 5]
"""
import os, sys, csv, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import soundfile as sf
import torch
from scipy.spatial.transform import Rotation
from speech_to_motion_small import load_trained, generate, T_M
from speech_to_motion_aligned import read_window, HubertFeaturizer, FPS_M
from vqvae_motion_tokenizer import (VQVAETokenizer, Normalizer, preprocess_motion, features_to_smplx, smooth_features,
                                    load_smplx_sequence, heading_yaw)

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("tok"); ap.add_argument("out")
ap.add_argument("--sigma", type=float, default=-2.5); ap.add_argument("--dur", type=float, default=60.0)
ap.add_argument("--hop", type=float, default=2.0); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--temp", type=float, default=1.3); ap.add_argument("--guid", type=float, default=3.5)
ap.add_argument("--data", default="outputs/stream-hubert-nov04")
ap.add_argument("--adapt", type=float, nargs=2, default=None, help="uncertainty-adaptive guidance g_min g_max (replaces --guid)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
model = load_trained(args.ckpt)[0]
vq = VQVAETokenizer(); vq.load(f"{args.tok}/tokenizer.pt"); vq.model.eval()
norm = Normalizer(); norm.load(f"{args.tok}/normalizer.npz")
feat = HubertFeaturizer()
L = model.L; S = T_M + L; DUR = S / FPS_M
NF = np.load(f"{args.data}/val_hubert.npy", mmap_mode="r").shape[1]
H = int(round(args.hop * FPS_M))                     # new motion slots per step
W = int(args.dur * 30)                               # output frames (30 fps)

# 20 val tracks long enough for the stream, distinct audio files, fixed pick
seen, tracks = set(), []
for w in csv.DictReader(open(f"{args.data}/val_windows.csv")):
    a = w["audio_filename"]
    if a in seen:
        continue
    seen.add(a)
    if sf.info(a).duration >= 1.0 + args.dur + DUR:
        tracks.append(a)
rng = np.random.default_rng(0); tracks = sorted(rng.choice(tracks, min(args.n, len(tracks)), replace=False))
print(f"{len(tracks)} tracks of >= {args.dur + DUR:.0f} s", flush=True)


def to_window(motion):
    """world-frame SMPL-X (T, 159) -> normalised features relative to its own first frame"""
    f = preprocess_motion(motion, norm.orient_repr, norm.pose_repr)
    return (f - norm.mean) / norm.std


def decode(ids):
    with torch.no_grad():
        f = vq.model.decode_from_tokens(torch.as_tensor(ids, device="cuda")[None].long())[0].permute(1, 0).float().cpu().numpy()
    f = smooth_features(f * norm.std + norm.mean, args.sigma)
    return features_to_smplx(f, norm.orient_repr, norm.pose_repr)[:, :159]


def to_world(m, ref):
    """window-frame motion (frame 0 facing +Z at the origin) -> world, continuing from world pose ref (159,)"""
    rot = Rotation.from_euler("y", heading_yaw(ref[:3]) if norm.orient_repr == "6d_rel" else 0.0)
    out = m.copy()
    out[:, :3] = (rot * Rotation.from_rotvec(m[:, :3])).as_rotvec()
    out[:, 156:159] = rot.apply(m[:, 156:159]) + ref[156:159]
    return out


def gen(feats, prefix=None):
    return generate(model, None, None, feats, do_sample=True, temperature=args.temp, top_p=1.0, prefix=prefix, guidance=args.guid,
                    adapt=tuple(args.adapt) if args.adapt else None)


def stream(audio, t0):
    """continuous: returns (W, 159) world motion and the boundary frames"""
    ids = gen(feat.stream(read_window(audio, t0, DUR), NF, S))
    out = decode(ids)                                            # (300, 159), first window: all emitted
    out[:, 156:159] -= out[0, 156:159]
    bounds = []
    k = 1
    while len(out) < W:
        s = k * H                                               # window start, in slots from t0
        f0 = 2 * s                                              # its first frame
        over = out[f0:f0 + 2 * (T_M - H)]                       # emitted motion inside the new window
        x = torch.as_tensor(to_window(over).T[None], device="cuda").float()
        with torch.no_grad():
            pre = vq.model.encode_to_tokens(x)[0].cpu().numpy()  # (T_M - H, Q)
        ids = gen(feat.stream(read_window(audio, t0 + s / FPS_M, DUR), NF, S), prefix=pre)
        win = to_world(decode(ids), over[0])
        new = win[2 * (T_M - H):].copy()
        # the re-decoded window does not end exactly where the emitted motion did: carry the difference at the seam
        # into the NEW frames only, fading over ~0.3 s (causal: emitted frames are never changed)
        d = out[-1, 3:159] - win[2 * (T_M - H) - 1, 3:159]
        new[:, 3:159] += d[None] * np.exp(-np.arange(len(new)) / 5.0)[:, None]
        bounds.append(len(out)); out = np.concatenate([out, new]); k += 1
    return out[:W], bounds


def reset(audio, t0):
    """baseline: independent unseeded 10 s windows, each placed where the last one ended"""
    out = None
    for k in range(int(np.ceil(W / (2 * T_M)))):
        m = decode(gen(feat.stream(read_window(audio, t0 + k * T_M / FPS_M, DUR), NF, S)))
        m[:, 156:159] -= m[0, 156:159]
        if out is None:
            out = m
        else:
            m = to_world(m, out[-1]); out = np.concatenate([out, m])
    return out[:W], list(range(2 * T_M, W, 2 * T_M))


import motion_dist_metrics as mdm                              # its PCA / speed features (built at import)


def jerk(m):
    return np.abs(np.diff(m[:, 3:66], 3, axis=0)).mean()


def jump(m, b):
    """mean pose change across window boundaries / mean pose change between ordinary frames"""
    d = np.abs(np.diff(m[:, 3:66], axis=0)).mean(1)
    return float(np.mean([d[i - 1] for i in b]) / (d.mean() + 1e-8)) if b else float("nan")


res = {"stream": [], "reset": []}
for ti, a in enumerate(tracks):
    t0 = 1.0
    real = load_smplx_sequence(os.path.dirname(os.path.dirname(a)))[round(t0 * 30):round(t0 * 30) + W]
    for name, fn in (("stream", stream), ("reset", reset)):
        m, b = fn(a, t0)
        np.save(f"{args.out}/track{ti}_{name}.npy", m)
        res[name].append((m, real, b))
    print(f"track {ti + 1}/{len(tracks)} done", flush=True)

for name, rows in res.items():
    segs = int(args.dur // 10)
    for k in range(segs):
        sl = slice(300 * k, 300 * (k + 1))
        g = np.concatenate([mdm.feats(m[sl]) for m, _, _ in rows]); r = np.concatenate([mdm.feats(re[sl]) for _, re, _ in rows])
        sp = np.mean([mdm.speeds(m[sl]).mean() / (mdm.speeds(re[sl]).mean() + 1e-8) for m, re, _ in rows])
        jr = np.mean([jerk(m[sl]) / (jerk(re[sl]) + 1e-8) for m, re, _ in rows])
        print(f"CONT mode={name} seg={k * 10}-{k * 10 + 10}s fd={mdm.fd(g, r):.2f} speed={sp:.2f} jerk={jr:.2f} n={len(rows)}", flush=True)
    print(f"CONT mode={name} boundary_jump={np.mean([jump(m, b) for m, _, b in rows]):.2f} (1 = like any other frame)", flush=True)
