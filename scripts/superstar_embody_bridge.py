"""Super Star outputs on Embody-3D (scripts/superstar_embody_eval.sh: SS_DUMP npz per window, made in the +Z-facing
frame of scripts/superstar_embody_data.py) -> our format, so motion_dist_metrics.py / audio_motion_match.py score
them like our models and EMAGE: <out>/val<i>_<subject>_<mode>.npy, (300, 159) axis-angle (orient, body, hands, transl),
turned back to the real starting heading and position (as scripts/emage_bridge.py). Shorter outputs are padded by
holding the last frame (counted and printed). Runs in the EMAGE venv.

usage: python scripts/superstar_embody_bridge.py <dump_dir> <out_dir> <mode> [--root /data2/home/mubtasim/ext/superstar_embody]
"""
import os, sys, csv, glob, argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("mode")
ap.add_argument("--root", default="/data2/home/mubtasim/ext/superstar_embody")
args = ap.parse_args()
args.dump, args.out = os.path.abspath(args.dump), os.path.abspath(args.out)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # windows.csv paths are repo-relative
PARTS = ["smplx_mesh_global_orient", "smplx_mesh_body_pose", "smplx_mesh_left_hand_pose",
         "smplx_mesh_right_hand_pose", "smplx_mesh_transl"]
W = 300


def load_seq(subj_dir):
    a = [np.load(glob.glob(f"{subj_dir}/{p}/*.npy")[0]).reshape(-1, d) for p, d in zip(PARTS, (3, 63, 45, 45, 3))]
    T = min(len(x) for x in a)
    return np.concatenate([x[:T] for x in a], 1).astype(np.float32)


def turn(m, yaw, origin):
    rot = R.from_euler("y", yaw)
    out = m.copy()
    out[:, :3] = (rot * R.from_rotvec(m[:, :3])).as_rotvec()
    out[:, 156:159] = rot.apply(m[:, 156:159] - origin) + origin
    return out


os.makedirs(args.out, exist_ok=True)
n_ok, short = 0, []
for r in csv.DictReader(open(f"{args.root}/windows.csv")):
    if r["split"] == "train":
        continue
    f = f"{args.dump}/{r['id']}_output.npz"
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True)
    aa, tr = d["poses"].astype(np.float32), d["trans"].astype(np.float32)
    T = min(len(aa), len(tr), W)
    if T < W:
        short.append((r["id"], T))
    aa = aa[:T].reshape(T, 55, 3); tr = tr[:T]
    pred = np.concatenate([aa[:, 0], aa[:, 1:22].reshape(T, 63), aa[:, 25:55].reshape(T, 90), tr], 1)
    pred = np.concatenate([pred, np.repeat(pred[-1:], W - T, 0)]) if T < W else pred
    sd = os.path.dirname(os.path.dirname(r["audio_filename"]))
    g = load_seq(sd)[round(float(r["start_sec"]) * 30)]
    fw = R.from_rotvec(g[:3]).apply([0, 0, 1.0]); yaw0 = np.arctan2(fw[0], fw[2])
    origin = g[156:159].copy()
    pred[:, 156:159] += origin * np.array([1, 0, 1], np.float32)          # data was x, z = 0 at frame 0
    np.save(f"{args.out}/val{r['val_index']}_{os.path.basename(sd)}_{args.mode}.npy", turn(pred, yaw0, origin))
    n_ok += 1
print(f"{n_ok} windows -> {args.out} (mode {args.mode}); short outputs padded: {short[:5]}{' ...' if len(short) > 5 else ''} ({len(short)})")
