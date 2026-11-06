"""Listener-response metric (Embody-3D conversation benchmark): does the motion react to the PARTNER's speech? The
speech-match metric scores motion against the person's own audio, so it cannot see this. For each eval window:
partner speech envelope e(t) (scripts/build_partner.py log-mel -> energy, 25 fps resampled to 30 fps) and head speed
v(t) (frame-to-frame change of the head joint rotation, 30 fps); score = max over lags 0..1 s of the correlation of
e(t) with v(t + lag) (head reacting after the partner speaks), averaged over windows. Reported for the real motion,
each eval mode, and a shuffled control (each motion against another window's partner) giving the no-reaction level.

usage: python scripts/listener_response.py <eval_dir> [data_dir=outputs/stream-hubert-nov04] [mode prefix=sample1.3p1f15g3.5]
"""
import os, sys, re, csv, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from vqvae_motion_tokenizer import load_smplx_sequence

ev = sys.argv[1]
data = sys.argv[2] if len(sys.argv) > 2 else "outputs/stream-hubert-nov04"
prefix = sys.argv[3] if len(sys.argv) > 3 else "sample1.3p1f15g3.5"
HEAD = 3 + (15 - 1) * 3                          # body pose (after 3 global orient) starts at joint 1; head = joint 15
va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
part = np.load(f"{data}/val_partner.npy", mmap_mode="r")
LAGS = range(0, 31)


def envelope(i, n):
    e = np.log(np.exp(np.asarray(part[i], np.float64)).sum(1) + 1e-6)           # (25 fps) log energy
    return np.interp(np.arange(n) / 30.0, np.arange(len(e)) / 25.0, e)


def head_speed(m):
    return np.r_[0, np.linalg.norm(np.diff(m[:, HEAD:HEAD + 3], axis=0), axis=1)]


def score(e, v):
    best = -1.0
    for L in LAGS:
        a, b = e[:len(e) - L], v[L:]
        if a.std() > 1e-8 and b.std() > 1e-8:
            best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


files = {}
for p in glob.glob(f"{ev}/val*_{prefix}*.npy"):
    m = re.match(rf"val(\d+)_.*?_({re.escape(prefix)}(?:s\d+)?)\.npy$", os.path.basename(p))
    if m:
        files.setdefault(m.group(2), {})[int(m.group(1))] = p
idx = sorted(next(iter(files.values())))
real = {}
for i in idx:
    sm = load_smplx_sequence(os.path.dirname(os.path.dirname(va[i]["audio_filename"])))
    f0 = round(float(va[i]["start_sec"]) * 30); real[i] = sm[f0:f0 + 300]
rows = {"real": [score(envelope(i, len(real[i])), head_speed(real[i])) for i in idx]}
for mode, fs in sorted(files.items()):
    rows[mode] = [score(envelope(i, 300), head_speed(np.load(fs[i])[:300])) for i in idx]
    sh = idx[1:] + idx[:1]
    rows[mode + "_shuffled"] = [score(envelope(j, 300), head_speed(np.load(fs[i])[:300])) for i, j in zip(idx, sh)]
sh = idx[1:] + idx[:1]
rows["real_shuffled"] = [score(envelope(j, len(real[i])), head_speed(real[i])) for i, j in zip(idx, sh)]
for k, v in rows.items():
    print(f"LISTENRESP eval={os.path.basename(ev)} mode={k} r={np.mean(v):.3f} n={len(v)}", flush=True)
