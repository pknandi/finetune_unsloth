"""Signed speech-motion timing (O2): does the motion lead or lag the speech? People start gestures before the
speech they go with (McNeill's phonological synchrony rule; corpus studies), so an online model that only reacts
to audio it has heard should lag. Per window: the person's own loudness envelope a(t) and motion speed v(t)
(head, upper body, both hands; scripts/motion_dist_metrics.py), both 30 fps and high-passed (1 s moving average
removed); the correlation of a(t) with v(t + lag) for lags -1..+1 s, averaged over windows. Reports the lag of the
peak of the mean curve (negative = motion LEADS speech, positive = motion lags) with a bootstrap 95% interval over
windows, and the peak correlation. Real motion of the same windows is the reference.

usage: python scripts/lead_lag.py <eval_dir>[,<eval_dir2>...] [mode prefix=sample1.3p1f15g3.5] [data=outputs/run-full-q8-oct21/aligned_data]
"""
import os, sys, re, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import csv
import numpy as np
import motion_dist_metrics as mdm

evs = sys.argv[1].split(",")
prefix = sys.argv[2] if len(sys.argv) > 2 else "sample1.3p1f15g3.5"
data = sys.argv[3] if len(sys.argv) > 3 else "outputs/run-full-q8-oct21/aligned_data"
va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
LAGS = np.arange(-30, 31)


def curve(a, v):
    a, v = mdm.highpass(a), mdm.highpass(v)
    out = []
    for L in LAGS:
        x, y = (a[:-L], v[L:]) if L > 0 else (a[-L:], v[:L]) if L < 0 else (a, v)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-8 and y.std() > 1e-8 else np.nan)
    return np.array(out)


def summary(curves, name):
    C = np.array(curves)
    m = np.nanmean(C, 0); k = int(np.nanargmax(m))
    rng = np.random.default_rng(0)
    boots = [LAGS[int(np.nanargmax(np.nanmean(C[rng.integers(0, len(C), len(C))], 0)))] for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 1000 / 30
    print(f"LEADLAG {name} peak_lag_ms={LAGS[k] * 1000 / 30:+.0f} 95%CI=[{lo:+.0f},{hi:+.0f}] peak_r={m[k]:.3f} "
          f"r_at0={m[30]:.3f} n={len(C)}", flush=True)


files = {}
for ev in evs:
    for p in glob.glob(f"{ev}/val*_{prefix}*.npy"):
        m = re.match(rf"val(\d+)_.*?_({re.escape(prefix)}(?:s\d+)?)\.npy$", os.path.basename(p))
        if m and not p.endswith("_ids.npy"):
            files.setdefault(int(m.group(1)), []).append(p)
idx = sorted(files)
real, gen = [], []
for i in idx:
    a_path, s = va[i]["audio_filename"], float(va[i]["start_sec"])
    loud = mdm.loudness(a_path, s, mdm.WIN)
    g = mdm.gt_window(a_path, s)
    real.append(curve(loud, mdm.speeds(g).sum(1)[:len(loud) - 1]))
    for p in files[i]:
        gen.append(curve(loud, mdm.speeds(np.load(p)[:mdm.WIN]).sum(1)[:len(loud) - 1]))
summary(real, "real")
summary(gen, f"{os.path.basename(evs[0])}:{prefix} ({len(gen) // max(1, len(idx))} draws)")
