"""Benchmark baseline: audio nearest-neighbour retrieval (online variant). For each evaluation window, the training
window whose streamed (causal) speech features are most similar (cosine of the time-averaged HuBERT features) is
found, and its real motion is returned, placed at the evaluation window's starting heading. The output is real human
motion, so it measures how far "plausible motion for similar-sounding speech" gets without any timing alignment.
Writes an eval dir in scripts/aligned_eval.py's layout (val<i>_<subject>_<mode>.npy, (300, 159)), scored with
scripts/motion_dist_metrics.py and scripts/audio_motion_match.py like every model.

usage: python scripts/nn_retrieval.py <out_dir> [--group k] [--data outputs/stream-hubert-nov04] [--train outputs/stream-f24-nov04]
"""
import os, sys, csv, random, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from scipy.spatial.transform import Rotation
from vqvae_motion_tokenizer import load_smplx_sequence, heading_yaw

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--group", type=int, default=0); ap.add_argument("--n", type=int, default=60)
ap.add_argument("--data", default="outputs/stream-hubert-nov04"); ap.add_argument("--train", default="outputs/stream-f24-nov04")
ap.add_argument("--mode", default="nnret")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
va = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))
random.seed(0)
pick = sorted(random.sample(range(len(va)), min(args.n, len(va))))
if args.group:                                     # the same disjoint groups as scripts/aligned_eval.py EVAL_GROUP
    rest = [i for i in np.random.RandomState(0).permutation(len(va)) if i not in set(pick)]
    pick = sorted(rest[(args.group - 1) * args.n:args.group * args.n])
tr = list(csv.DictReader(open(f"{args.train}/train_windows.csv")))
trf = np.load(f"{args.train}/train_hubert.npy", mmap_mode="r")
vaf = np.load(f"{args.data}/val_hubert.npy", mmap_mode="r")
emb = lambda x: (lambda m: m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8))(np.asarray(x, np.float32).mean(-2))
TR = np.concatenate([emb(trf[i:i + 2000]) for i in range(0, len(trf), 2000)])
vsess = lambda w: w["audio_filename"].split("/")[-4]
for i in pick:
    q = emb(vaf[i][None])[0]
    order = np.argsort(-(TR @ q))
    w = va[i]
    for j in order:                                # nearest training window (train and val sessions are disjoint)
        src = tr[j]
        m = load_smplx_sequence(os.path.dirname(os.path.dirname(src["audio_filename"])))
        f0 = round(float(src["start_sec"]) * 30)
        if len(m) >= f0 + 300:
            m = m[f0:f0 + 300].copy(); break
    # place: translation relative to its start, heading turned to the eval window's real starting heading
    m[:, 156:159] -= m[0, 156:159]
    g0 = load_smplx_sequence(os.path.dirname(os.path.dirname(w["audio_filename"])))[round(float(w["start_sec"]) * 30)]
    rot = Rotation.from_euler("y", heading_yaw(g0[:3]) - heading_yaw(m[0, :3]))
    m[:, :3] = (rot * Rotation.from_rotvec(m[:, :3])).as_rotvec(); m[:, 156:159] = rot.apply(m[:, 156:159])
    subj = os.path.basename(os.path.dirname(os.path.dirname(w["audio_filename"])))
    np.save(f"{args.out}/val{i}_{subj}_{args.mode}.npy", m[:, :159].astype(np.float32))
print(f"wrote {len(pick)} windows to {args.out}", flush=True)
