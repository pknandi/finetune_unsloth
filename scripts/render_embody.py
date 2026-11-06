"""Mesh clips of Embody-3D held-out windows (scripts/aligned_eval.py outputs) for figures and the user study,
rendered exactly as the BEAT2 clips (scripts/mesh_render.py). <pred> is an aligned_eval output dir and <mode> the
file tag (e.g. sample1.3p1f15g3.5), or "gt" for the real motion. The clip index is the val window index in the
file names (val<i>_<subject>_<mode>.npy, windows from <data_dir>/val_windows.csv). Runs in the Super Star venv.

usage: python scripts/render_embody.py <data_dir> <i> <pred_dir|gt> <mode> <out.mp4> [--sec 10]
"""
import os, sys, csv, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_render import vertices, rest, camera, render, FPS

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("i", type=int); ap.add_argument("pred"); ap.add_argument("mode"); ap.add_argument("out")
ap.add_argument("--sec", type=float, default=10.0)
args = ap.parse_args()
w = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))[args.i]
audio, start = w["audio_filename"], float(w["start_sec"])
sd = os.path.dirname(os.path.dirname(audio))
part = lambda p: np.load(glob.glob(f"{sd}/smplx_mesh_{p}/*.npy")[0]).astype(np.float32)
n = round(args.sec * FPS); f0 = round(start * FPS)
gt = np.concatenate([part(p).reshape(len(part(p)), -1) for p in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")], 1)[f0:f0 + n]
betas = part("betas").reshape(-1)
if args.pred == "gt":
    p = gt
else:
    p = np.load(f"{args.pred}/val{args.i}_{os.path.basename(sd)}_{args.mode}.npy").astype(np.float32)[:n, :156]
assert len(p) == n and len(gt) == n, (len(p), len(gt), n)
vg, faces = vertices(betas, gt[:, :3], gt[:, 3:66], gt[:, 66:111], gt[:, 111:156])
v = vg if args.pred == "gt" else vertices(betas, p[:, :3], p[:, 3:66], p[:, 66:111], p[:, 111:156])[0]
render(v, faces, camera(rest(betas), gt[0, :3]), audio, start, args.sec, args.out)
print(f"{args.out}: {n} frames", flush=True)
