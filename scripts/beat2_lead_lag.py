"""BEAT2 signed speech-motion timing (2026-11-12), the BEAT2 counterpart of scripts/lead_lag.py: per test recording,
cut into 10 s windows; per window the speaker's loudness envelope a(t) and motion speed v(t) (head, upper body, both
hands, from the SMPL-X axis-angle pose; scripts/motion_dist_metrics.py), 30 fps, high-passed; correlation of a(t)
with v(t + lag) for lags -1..+1 s, averaged over windows. Prints the lag of the peak of the mean curve (negative =
motion LEADS speech, positive = motion lags) with a bootstrap 95% interval over windows. Real motion is the reference.
Several pred dirs (e.g. the 6 test draws of one system) are pooled.
NOTE 2026-11-12: on BEAT2 the coupling is too weak to locate the peak (real r 0.046, CI about +-500 ms); not reportable.
usage: python scripts/beat2_lead_lag.py <pred_dir>[,<pred_dir2>...] [name]
"""
import os, sys, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import motion_dist_metrics as mdm
LAGS = np.arange(-30, 31)


def curve(a, v):          # as scripts/lead_lag.py
    a, v = mdm.highpass(a), mdm.highpass(v)
    out = []
    for L in LAGS:
        x, y = (a[:-L], v[L:]) if L > 0 else (a[-L:], v[:L]) if L < 0 else (a, v)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-8 and y.std() > 1e-8 else np.nan)
    return np.array(out)


def summary(curves, name):
    C = np.array(curves); m = np.nanmean(C, 0); k = int(np.nanargmax(m)); rng = np.random.default_rng(0)
    boots = [LAGS[int(np.nanargmax(np.nanmean(C[rng.integers(0, len(C), len(C))], 0)))] for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 1000 / 30
    print(f"B2LEADLAG {name} peak_lag_ms={LAGS[k] * 1000 / 30:+.0f} 95%CI=[{lo:+.0f},{hi:+.0f}] peak_r={m[k]:.3f} "
          f"r_at0={m[30]:.3f} n={len(C)}", flush=True)


B = "/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0"
R = "outputs/beat2-spk2-oct30/split/recordings.csv"
preds = sys.argv[1].split(","); name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(preds[0].rstrip("/"))
ids = [r["id"] for r in csv.DictReader(open(R)) if r["split"] == "test"]
to156 = lambda p: np.concatenate([p[:, :66], p[:, 75:165]], 1)     # drop jaw / eyes: (root, body, lhand, rhand)

real, gen = [], []
for rid in ids:
    gt = to156(np.load(f"{B}/smplxflame_30/{rid}.npz", allow_pickle=True)["poses"])
    P = [to156(np.load(f"{pd}/{rid}_output.npz", allow_pickle=True)["poses"]) for pd in preds]
    T = min([len(gt)] + [len(p) for p in P])
    for s in range(0, T - mdm.WIN + 1, mdm.WIN):
        loud = mdm.loudness(f"{B}/wave16k/{rid}.wav", s / mdm.FPS, mdm.WIN)
        if len(loud) < mdm.WIN:
            continue
        real.append(curve(loud, mdm.speeds(gt[s:s + mdm.WIN]).sum(1)))
        for p in P:
            gen.append(curve(loud, mdm.speeds(p[s:s + mdm.WIN]).sum(1)))
summary(real, "real")
summary(gen, f"{name} ({len(preds)} dirs)")
