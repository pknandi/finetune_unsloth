"""Mesh clips for the user study / supplementary video: one BEAT2 recording segment of one method, rendered by
scripts/mesh_render.py (SMPL-X neutral 2020, the subject's betas, no translation, fixed camera from the rest pose).
Jaw and eyes come from the real motion for every method (as in scripts/beat2_score.py --jaw gt), so raters judge
the body and hand motion only. <pred> is a <dir> holding <id>_output.npz (poses (T,165) at 30 fps), or "gt".
Needs LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl (scripts/render_clips.sh sets it).
Runs in the Super Star venv. usage: python scripts/render_beat2.py <pred> <id> <start_sec> <seconds> <out.mp4>
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_render import vertices, rest, camera, render, FPS

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("rid"); ap.add_argument("start", type=float); ap.add_argument("sec", type=float)
ap.add_argument("out")
ap.add_argument("--beat2", default="/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0")
args = ap.parse_args()
g = np.load(f"{args.beat2}/smplxflame_30/{args.rid}.npz", allow_pickle=True)
p = g["poses"] if args.pred == "gt" else np.load(f"{args.pred}/{args.rid}_output.npz", allow_pickle=True)["poses"]
f0, n = round(args.start * FPS), round(args.sec * FPS)
assert len(p) >= f0 + n and len(g["poses"]) >= f0 + n, (len(p), len(g["poses"]), f0 + n)
p = p[f0:f0 + n].astype(np.float32).copy()
gp = g["poses"][f0:f0 + n].astype(np.float32)
expr = g["expressions"][f0:f0 + n].astype(np.float32) if "expressions" in g.files else None
betas = g["betas"][:10].astype(np.float32)
v, faces = vertices(betas, p[:, :3], p[:, 3:66], p[:, 75:120], p[:, 120:165], jaw=gp[:, 66:69], eyes=gp[:, 69:75], expr=expr)
render(v, faces, camera(rest(betas)), f"{args.beat2}/wave16k/{args.rid}.wav", args.start, args.sec, args.out)
print(f"{args.out}: {n} frames", flush=True)
