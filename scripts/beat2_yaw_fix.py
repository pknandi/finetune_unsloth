"""Heading-drift correction for our BEAT2 outputs: generation chains 10 s windows, each relative to the previous
window's heading, so small turns add up (mean |yaw - real| 44 deg for F2 vs EMAGE 16, LoM 21). This pulls the
root yaw back toward the starting heading (the seed frame's, which our model already gets): drift d(t) = causal
exponential average of (yaw(t) - yaw(0)) with time constant <tau> s, and the root (and translation, about the
start position) is turned by -d(t) about the vertical. Uses only our own output (no real motion), causal, so the
online model can use it too. tau 0 = remove all yaw change (heading fixed at the start).
--anchor <beat2 dir>: pull toward the real first frame's heading instead of our own first frame's (our output
starts up to ~25 deg off it; every draw is already placed at that real first-frame heading, so no new input).
usage: python scripts/beat2_yaw_fix.py <in_dir> <out_dir> <tau_sec> [--anchor <beat2 dir>]
"""
import os, sys, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

src, out, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
anchor = sys.argv[sys.argv.index("--anchor") + 1] if "--anchor" in sys.argv else None
os.makedirs(out, exist_ok=True)
a = 0.0 if tau <= 0 else 1.0 - np.exp(-1.0 / (30 * tau))
for f in sorted(glob.glob(f"{src}/*_output.npz")):
    d = dict(np.load(f, allow_pickle=True))
    p = d["poses"].astype(np.float64).copy()
    fw = R.from_rotvec(p[:, :3]).apply([0, 0, 1.0])
    y0 = np.arctan2(fw[0, 0], fw[0, 2])
    if anchor:
        g0 = np.load(f"{anchor}/smplxflame_30/{os.path.basename(f)[:-len('_output.npz')]}.npz")["poses"][0, :3]
        fg = R.from_rotvec(g0).apply([0, 0, 1.0]); y0 = np.arctan2(fg[0], fg[2])
    y = np.unwrap(np.arctan2(fw[:, 0], fw[:, 2]))
    y = y - y0 - 2 * np.pi * np.round((y[0] - y0) / (2 * np.pi))
    if tau <= 0:
        dr = y
    else:
        dr = np.zeros_like(y)
        for t in range(1, len(y)):
            dr[t] = dr[t - 1] + a * (y[t] - dr[t - 1])
    fix = R.from_euler("y", -dr)
    p[:, :3] = (fix * R.from_rotvec(p[:, :3])).as_rotvec()
    d["poses"] = p.astype(d["poses"].dtype)
    if "trans" in d and len(d["trans"]) == len(p):
        tr = d["trans"].astype(np.float64)
        d["trans"] = (fix.apply(tr - tr[0]) + tr[0]).astype(d["trans"].dtype)
    np.savez(f"{out}/{os.path.basename(f)}", **d)
print(f"{out}: tau {tau}")
