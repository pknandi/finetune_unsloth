"""Does the one-sided output smoothing cause the online models' lag? (2026-11-12) The half-Gaussian over past frames
(sigma -2.5, 30 fps) delays the motion by sum(i w_i) = ~2 frames (67 ms). For an eval dir's saved token ids
(val*_<mode>*_ids.npy), re-decode with other render sigmas (causal decoder, same tokens) and report the lead/lag
(scripts/lead_lag.py: peak of the mean loudness-speed cross-correlation, bootstrap CI over windows) and the median
jerk ratio of the body speed vs the real motion's.
usage: python scripts/lag_resmooth.py <eval_dir>[,...] <mode prefix> <tok_dir> <sigmas e.g. -2.5,-1,0>
"""
import os, sys, re, glob, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import motion_dist_metrics as mdm
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, features_to_smplx, smooth_features

evs, prefix, tok, sigmas = sys.argv[1].split(","), sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4].split(",")]
va = list(csv.DictReader(open("outputs/run-full-q8-oct21/aligned_data/val_windows.csv")))
LAGS = np.arange(-30, 31)


def curve(a, v):
    a, v = mdm.highpass(a), mdm.highpass(v); r = []
    for L in LAGS:
        x, y = (a[:-L], v[L:]) if L > 0 else (a[-L:], v[:L]) if L < 0 else (a, v)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        r.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-8 and y.std() > 1e-8 else np.nan)
    return np.array(r)


files = []
for d in evs:
    for p in glob.glob(f"{d}/val*_{prefix}*_ids.npy"):
        m = re.match(rf"val(\d+)_.*?_{re.escape(prefix)}(?:s\d+)?_ids\.npy$", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
print(f"{len(files)} token files", flush=True)
vq = VQVAETokenizer(); vq.load(f"{tok}/tokenizer.pt"); nm = Normalizer(); nm.load(f"{tok}/normalizer.npz")
feats = {p: vq.decode(np.load(p)) * nm.std + nm.mean for _, p in files}      # decoded once; only the render smoothing varies
loud = {}; gtj = {}
for sg in sigmas:
    C, J = [], []
    for i, p in files:
        if i not in loud:
            loud[i] = mdm.loudness(va[i]["audio_filename"], float(va[i]["start_sec"]), mdm.WIN)
            g = mdm.gt_window(va[i]["audio_filename"], float(va[i]["start_sec"]))
            gtj[i] = np.abs(np.diff(mdm.speeds(g).sum(1), 2)).mean() if len(g) == mdm.WIN else np.nan
        m = features_to_smplx(smooth_features(feats[p], sg), nm.orient_repr, nm.pose_repr)[:mdm.WIN]   # pose columns as motion_ids_to_smplx
        v = mdm.speeds(m).sum(1); l = loud[i]
        C.append(curve(l, v[:len(l) - 1])); J.append(np.abs(np.diff(v, 2)).mean() / gtj[i])
    C = np.array(C); mc = np.nanmean(C, 0); k = int(np.nanargmax(mc)); rng = np.random.default_rng(0)
    boots = [LAGS[int(np.nanargmax(np.nanmean(C[rng.integers(0, len(C), len(C))], 0)))] for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 1000 / 30
    print(f"RESMOOTH sigma={sg} peak_lag_ms={LAGS[k] * 1000 / 30:+.0f} 95%CI=[{lo:+.0f},{hi:+.0f}] peak_r={mc[k]:.3f} "
          f"jerk_ratio_speed={np.nanmedian(J):.2f} n={len(C)}", flush=True)
