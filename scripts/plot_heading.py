"""Heading (facing direction) over time on BEAT2 test, per method: one recording's curves + the mean absolute
difference to the real speaker's heading over all 15 test recordings (printed and in the legend).
usage: python scripts/plot_heading.py <rec id> <out_prefix>   (emage venv)
"""
import sys, csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0"; BB = "outputs/beat2-bench"
rid, out = sys.argv[1:3]
M = [("Real", None, "k"), ("Ours (no heading fix)", "ft_f2hw1.0_t1g2_f15s0", "tab:red"),
     ("Ours (heading fix)", "ft_f2hw1.0_t1g2_f15s0_yaw2a", "tab:blue"),
     ("EMAGE", "emage_pre_fc", "tab:green"), ("LoM", "lom_released_test", "tab:purple")]
ids = [r["id"] for r in csv.DictReader(open("outputs/beat2-spk2-oct30/split/recordings.csv")) if r["split"] == "test"]
yaw = lambda p: np.degrees(np.unwrap(np.arctan2(*R.from_rotvec(p[:, :3]).apply([0, 0, 1.0])[:, [0, 2]].T)))
load = lambda d, i: np.load(f"{B}/smplxflame_30/{i}.npz")["poses"] if d is None else np.load(f"{BB}/{d}/{i}_output.npz")["poses"]
fig, ax = plt.subplots(figsize=(7, 2.6))
for name, d, c in M:
    errs = []
    for i in ids:
        g, p = load(None, i), load(d, i); n = min(len(g), len(p))
        errs.append(np.abs((yaw(p[:n]) - yaw(g[:n]) + 180) % 360 - 180).mean())
    p = load(d, rid); t = np.arange(len(p)) / 30
    lab = name if d is None else f"{name}: {np.mean(errs):.0f} deg"
    ax.plot(t, yaw(p), c, lw=1.6 if d is None else 1.0, label=lab)
    print(f"{name}: mean |heading - real| {np.mean(errs):.1f} deg over {len(ids)} test recordings")
ax.set_xlabel("time (s)"); ax.set_ylabel("heading (deg)")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="mean difference to real\n(15 test recordings)", title_fontsize=7)
ax.grid(alpha=0.3)
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
