"""Reaction latency (benchmark metric): how long after a person starts to speak does their motion start? Speech onsets
are found in the person's own loudness envelope (30 fps, scripts/motion_dist_metrics.loudness): the first frame above
a per-window threshold (median + 0.5 x (90th pct - median)) after at least 0.5 s below it. Motion onset: the first
of a burst (3 frames in a row) between 0.5 s before and 1 s after the onset where upper-body + hands speed exceeds
its level 1.0-0.5 s before the onset by 3 standard deviations. Latency = motion onset - speech onset (negative: motion starts before the speech, as in people's
anticipatory gestures). Reports the median over all onsets and the share of onsets with any motion response within
1 s, for real motion and for each eval mode; an online model that only reacts to heard audio should be late.

usage: python scripts/reaction_latency.py <eval_dir>[,<dir2>...] [mode prefix=sample1.3p1f15g3.5] [data=outputs/run-full-q8-oct21/aligned_data]
"""
import os, sys, re, glob, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import motion_dist_metrics as mdm

evs = sys.argv[1].split(",")
prefix = sys.argv[2] if len(sys.argv) > 2 else "sample1.3p1f15g3.5"
data = sys.argv[3] if len(sys.argv) > 3 else "outputs/run-full-q8-oct21/aligned_data"
va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
FPS, SIL, BACK, WIN = 30, 15, 30, 30


def onsets(loud):
    thr = np.median(loud) + 0.5 * (np.percentile(loud, 90) - np.median(loud))
    on = loud > thr
    return [t for t in range(BACK, len(on) - WIN) if on[t] and not on[t - SIL:t].any()]


def latency(speed, t):
    if t < BACK:
        return None
    pre = speed[t - BACK:t - SIL]                                   # 1.0 .. 0.5 s before the speech onset
    thr = pre.mean() + 3 * pre.std() + 1e-6
    above = speed[t - SIL:t + WIN] > thr                            # search 0.5 s before .. 1 s after
    run = np.convolve(above.astype(int), np.ones(3, int), "valid") == 3   # a burst: 3 frames in a row
    hit = np.where(run)[0]
    return (hit[0] - SIL) / FPS if len(hit) else None


def summary(name, lats, n_on):
    got = [x for x in lats if x is not None]
    print(f"REACT {name} median_latency_ms={np.median(got) * 1000 if got else float('nan'):+.0f} "
          f"responded={len(got) / max(1, n_on):.2f} onsets={n_on}", flush=True)


files = {}
for ev in evs:
    for p in glob.glob(f"{ev}/val*_{prefix}*.npy"):
        m = re.match(rf"val(\d+)_.*?_({re.escape(prefix)}(?:s\d+)?)\.npy$", os.path.basename(p))
        if m and not p.endswith("_ids.npy"):
            files.setdefault(int(m.group(1)), []).append(p)
real, gen, n_real, n_gen = [], [], 0, 0
for i, ps in sorted(files.items()):
    a, s = va[i]["audio_filename"], float(va[i]["start_sec"])
    loud = mdm.loudness(a, s, mdm.WIN)
    ons = onsets(loud)
    g = mdm.gt_window(a, s)
    sp = mdm.speeds(g)[:, 1:].sum(1)
    real += [latency(sp, t) for t in ons]; n_real += len(ons)
    for p in ps:
        spg = mdm.speeds(np.load(p)[:mdm.WIN])[:, 1:].sum(1)
        gen += [latency(spg, t) for t in ons]; n_gen += len(ons)
summary("real", real, n_real)
summary(f"{os.path.basename(evs[0])}:{prefix}", gen, n_gen)
