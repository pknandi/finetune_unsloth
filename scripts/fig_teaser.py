"""Paper teaser figure (R1): what happens around the moment a person starts to speak. Over every speech onset (after
>= 0.5 s of silence, as scripts/reaction_latency.py) in the fresh Embody-3D evaluation groups 1-4, the person's
loudness and the motion speed (upper body + hands, z-scored per window) of real motion, the zero-look-ahead online
base and UpBeat are averaged, aligned at the onset (all samples, no clip selection). The time each curve reaches half
of its rise (from its pre-onset level to its peak after the onset) is marked.

usage: python scripts/fig_teaser.py [out=outputs/figures-paper/teaser.pdf] [upbeat eval dirs prefix mode]
"""
import os, sys, re, glob, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import motion_dist_metrics as mdm
plt.rcParams["font.family"] = "DejaVu Sans"

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/figures-paper/teaser.pdf"
F = "outputs/fix-oct21"
UP = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["e60_dt_Db_adapt"] + [f"e60_dt_Db_adapt_grp{k}" for k in (2, 3, 4)])
UPM = sys.argv[3] if len(sys.argv) > 3 else "sample1.3p1f15a1.5-5"
BASE = [f"e60_dt_v2_base_grp{k}" for k in range(1, 8)]; BM = "sample1.3p1f15g3.5"   # groups 1-7, corrected features (2026-11-12)
va = list(csv.DictReader(open("outputs/run-full-q8-oct21/aligned_data/val_windows.csv")))
FPS, SIL, BACK, WIN = 30, 15, 30, 30


def files(dirs, mode):
    out = {}
    for d in dirs:
        for p in glob.glob(f"{F}/{d}/val*_{mode}.npy"):                       # first draw only
            out[int(re.match(r"val(\d+)_", os.path.basename(p)).group(1))] = p
    return out


def onsets(l):
    thr = np.median(l) + 0.5 * (np.percentile(l, 90) - np.median(l)); on = l > thr
    return [t for t in range(BACK, len(on) - WIN) if on[t] and not on[t - SIL:t].any()]


def rise(sp, t):
    pre = sp[t - BACK:t - SIL]; thr = pre.mean() + 3 * pre.std() + 1e-6
    run = np.convolve((sp[t - SIL:t + WIN] > thr).astype(int), np.ones(3, int), "valid") == 3
    h = np.where(run)[0]
    return (h[0] - SIL) / FPS if len(h) else None


speed = lambda m: mdm.speeds(m[:mdm.WIN])[:, 1:].sum(1)                     # upper body + hands
z = lambda v: (v - v.mean()) / (v.std() + 1e-8)
A, Z = 45, 60                                                                # 1.5 s before, 2 s after the onset
U, B = files(UP, UPM), files(BASE, BM)
acc = {"loud": [], "real": [], "base": [], "up": []}
for i in sorted(set(U) & set(B)):
    w = va[i]; l = mdm.loudness(w["audio_filename"], float(w["start_sec"]), mdm.WIN)
    r, u, b = z(speed(mdm.gt_window(w["audio_filename"], float(w["start_sec"])))), z(speed(np.load(U[i]))), z(speed(np.load(B[i])))
    for t in onsets(l):
        if t - A < 0 or t + Z > len(r):
            continue
        acc["loud"].append(z(l)[t - A:t + Z]); acc["real"].append(r[t - A:t + Z]); acc["base"].append(b[t - A:t + Z]); acc["up"].append(u[t - A:t + Z])
M = {k: np.mean(v, 0) for k, v in acc.items()}
x = (np.arange(-A, Z)) / FPS * 1000


def half(c):
    lo = c[:A - 15].mean(); k = A + int(np.argmax(c[A:A + 30])); hi = c[k]
    j = np.where(c[A - 15:k + 1] >= lo + 0.5 * (hi - lo))[0]
    return x[A - 15 + j[0]] if len(j) else np.nan


print(f"onsets {len(acc['real'])} over {len(set(U) & set(B))} windows", flush=True)
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
ax0.fill_between(x, M["loud"].min(), M["loud"], color="#C9C9C9", lw=0); ax0.set_ylabel("speech\nloudness", fontsize=6.5)
ax0.set_yticks([]); ax0.axvline(0, color="#555555", lw=0.8, ls=":")
for k, lab, col in (("real", "Real motion", "#3A3A3A"), ("base", "Online base (zero look-ahead)", "#9E9E9E"),
                    ("up", "UpBeat (zero look-ahead)", "#138D7F")):
    h = half(M[k]); print(f"{lab}: half-rise at {h:+.0f} ms; rise (peak z within 1 s after - pre-onset mean) {M[k][A:A + 30].max() - M[k][:A].mean():.2f}", flush=True)
    ax1.plot(x, M[k], color=col, lw=1.4, label=f"{lab}: {h:+.0f} ms")
    ax1.plot([h], [np.interp(h, x, M[k])], "o", color=col, ms=3.2)
ax1.axvline(0, color="#555555", lw=0.8, ls=":")
ax1.set_ylabel("motion speed\n(z-score)", fontsize=6.5); ax1.tick_params(labelsize=6.5)
ax1.set_xlabel("time relative to speech onset (ms)", fontsize=7)
ax1.legend(fontsize=5.6, frameon=False, loc="upper left", title="half-rise time", title_fontsize=5.6)
for ax in (ax0, ax1):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=220)
print("wrote", out, flush=True)
