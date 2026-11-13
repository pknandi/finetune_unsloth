"""Paper figure (R2): speech-motion lead/lag curves. For each system, the mean (over windows and samples) correlation
between the person's loudness and motion speed (both high-passed, scripts/lead_lag.py) as a function of lag, scaled to
its own peak so that the timing, not the coupling strength, is compared; the peak is marked. Negative lag = motion
leads speech. Real motion over all 531 validation windows; models on the fresh evaluation groups 1-7 (2026-11-07).

usage: python scripts/fig_leadlag.py [out=outputs/figures-paper/leadlag.pdf]
"""
import os, sys, re, glob, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"            # the paper's Figure 1 style (scripts/fig_overview.py)
import motion_dist_metrics as mdm

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/figures-paper/leadlag.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
F = "outputs/fix-oct21"
va = list(csv.DictReader(open("outputs/run-full-q8-oct21/aligned_data/val_windows.csv")))
LAGS = np.arange(-30, 31)


def curve(a, v):
    a, v = mdm.highpass(a), mdm.highpass(v)
    r = []
    for L in LAGS:
        x, y = (a[:-L], v[L:]) if L > 0 else (a[-L:], v[:L]) if L < 0 else (a, v)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        r.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-8 and y.std() > 1e-8 else np.nan)
    return np.array(r)


_loud = {}


def loud(i):
    if i not in _loud:
        _loud[i] = mdm.loudness(va[i]["audio_filename"], float(va[i]["start_sec"]), mdm.WIN)
    return _loud[i]


def system(dirs, prefix):
    cs = []
    for d in dirs:
        for p in glob.glob(f"{F}/{d}/val*_{prefix}*.npy"):
            m = re.match(rf"val(\d+)_.*?_{re.escape(prefix)}(?:s\d+)?\.npy$", os.path.basename(p))
            if m and not p.endswith("_ids.npy"):
                i = int(m.group(1)); l = loud(i)
                cs.append(curve(l, mdm.speeds(np.load(p)[:mdm.WIN]).sum(1)[:len(l) - 1]))
    return np.nanmean(cs, 0)


real = []
for i, w in enumerate(va):
    g = mdm.gt_window(w["audio_filename"], float(w["start_sec"]))
    if len(g) == mdm.WIN:
        l = loud(i); real.append(curve(l, mdm.speeds(g).sum(1)[:len(l) - 1]))
G = lambda base: [f"{base}{k}" for k in (1, 2, 3, 4, 5, 6, 7)]
rows = [("Real motion", np.nanmean(real, 0), "#3A3A3A", "-"),
        ("Offline teacher (full look-ahead)", system(G("e60_dt_F2_grp") + G("e60_dt_F2_more_grp"), "sample1.3p1f15g3.5"), "#E07B39", "--"),
        ("Online base (zero look-ahead)", system(G("e60_dt_v2_base_grp"), "sample1.3p1f15g3.5"), "#9E9E9E", "-"),   # 2026-11-12: corrected features
        ("UpBeat (zero look-ahead)", system(G("e60_dt_v2_pas10ad_grp"), "sample1.3p1f15a1.5-5"), "#138D7F", "-")]   # 2026-11-13: final UpBeat (predict-ahead soup10 + adaptive), 6 samples
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ms = LAGS * 1000 / 30
for name, c, col, ls in rows:
    k = int(np.nanargmax(c)); y = c / c[k]
    ax.plot(ms, y, color=col, ls=ls, lw=1.4, label=f"{name}: {ms[k]:+.0f} ms")
    ax.plot([ms[k]], [1.0], "o", color=col, ms=3.5)
ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
ax.set_xlim(-600, 600); ax.set_ylim(-0.8, 1.08)
ax.set_xlabel("lag of motion relative to speech (ms)\nnegative: motion leads", fontsize=7)
ax.set_ylabel("loudness-speed correlation\n(relative to peak)", fontsize=7)
ax.tick_params(labelsize=6.5); ax.legend(fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=220)
print("wrote", out, [(r[0], f"{ms[int(np.nanargmax(r[1]))]:+.0f}") for r in rows], flush=True)
