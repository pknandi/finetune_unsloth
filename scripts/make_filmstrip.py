"""Paper figure: rows = methods, columns = moments of one rendered BEAT2 segment (scripts/render_clips.sh), the
speech waveform on top with the moments marked. Frames cropped to the body. Saves PDF + PNG.
usage: python scripts/make_filmstrip.py <rec id> <out_prefix> [--t 1,2.5,4,5.5,7] [--clip_start 20]
"""
import os, argparse, subprocess, tempfile
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("rid"); ap.add_argument("out")
ap.add_argument("--t", default="1,2.5,4,5.5,7"); ap.add_argument("--clip_start", type=float, default=20)
ap.add_argument("--dir", default="outputs/userstudy-nov04/clips")
ap.add_argument("--beat2", default="/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0")
args = ap.parse_args()
FF = "/data/home/mubtasim/conda-envs/motion/bin/ffmpeg"
ROWS = [("gt", "Real"), ("ours_f2", "Ours (offline)"), ("ours_ol", "Ours (online)"), ("emage", "EMAGE"), ("lom", "LoM")]
ts = [float(x) for x in args.t.split(",")]
tmp = tempfile.mkdtemp()
img = {}
for m, _ in ROWS:
    for j, t in enumerate(ts):
        f = f"{tmp}/{m}_{j}.png"
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", f"{args.dir}/{args.rid}_{m}.mp4", "-frames:v", "1", f],
                       check=True)
        img[m, j] = imageio.imread(f)[:, 96:416]          # 512 wide: keep the central 320 px (the body)
wav, sr = sf.read(f"{args.beat2}/wave16k/{args.rid}.wav")
seg = wav[int(args.clip_start * sr):int((args.clip_start + max(ts) + 1) * sr)]
nr, nc = len(ROWS), len(ts)
fig = plt.figure(figsize=(1.45 * nc + 1.0, 2.25 * nr + 0.7))
gs = fig.add_gridspec(nr + 1, nc + 1, height_ratios=[0.3] + [1] * nr, width_ratios=[0.45] + [1] * nc, hspace=0.04, wspace=0.02)
ax = fig.add_subplot(gs[0, 1:])
x = np.arange(len(seg)) / sr
ax.plot(x, seg, lw=0.4, color="0.35"); ax.set_xlim(0, max(ts) + 1); ax.axis("off")
for t in ts:
    ax.axvline(t, color="tab:red", lw=0.8)
for i, (m, name) in enumerate(ROWS):
    a = fig.add_subplot(gs[i + 1, 0]); a.axis("off")
    a.text(0.95, 0.5, name, ha="right", va="center", fontsize=10, rotation=90)
    for j in range(nc):
        a = fig.add_subplot(gs[i + 1, j + 1]); a.imshow(img[m, j]); a.axis("off")
        if i == 0:
            a.set_title(f"{ts[j]:.1f} s", fontsize=9)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"{args.out}.pdf / .png")
