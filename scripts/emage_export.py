"""Export the dense Embody-3D windows (scripts/build_dense.py) for EMAGE training (scripts/emage_train_vq.py,
scripts/emage_train_gen.py). Runs in the EMAGE venv. Each 10 s window is turned to face +Z at frame 0
and its x, z start at 0 (as scripts/emage_bridge.py does at test time).

usage: python scripts/emage_export.py <dense_dir> <out_dir>
Writes <out_dir>/{train,val}_pose.npy (N, 300, 156) float16 axis-angle (orient, body, hands; EMAGE adds
zero jaw/eyes), _trans.npy (N, 300, 3) float32, _contact.npy (N, 300, 4) uint8 (feet joints 7, 8, 10, 11
moving < 1 cm/frame, as PantoMatrix datasets/foot_contact.py), _audio.npy (N, 160000) int16 at 16 kHz,
_ok.npy (N,) bool (False: window short or unreadable, zeros stored). Row order = <dense_dir>/<split>_windows.csv.
"""
import os, sys, csv, glob
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import torch
import librosa
import smplx
from scipy.spatial.transform import Rotation as R

W, SR = 300, 16000
A = W * SR // 30
PARTS = ["smplx_mesh_global_orient", "smplx_mesh_body_pose", "smplx_mesh_left_hand_pose",
         "smplx_mesh_right_hand_pose", "smplx_mesh_transl"]
SMPLX_DIR = "/data2/home/mubtasim/ext/emage_work/hf/emage_evaltools/smplx_models"


def load_rec(job):
    """(subject dir, audio path, [(row, start_sec)]) -> [(row, pose, trans, audio, betas)] (pose None if unusable)"""
    sd, audio_path, items = job
    try:
        a = [np.load(glob.glob(f"{sd}/{p}/*.npy")[0]).reshape(-1, d) for p, d in zip(PARTS, (3, 63, 45, 45, 3))]
        T = min(len(x) for x in a)
        seq = np.concatenate([x[:T] for x in a], 1).astype(np.float32)
        betas = np.load(glob.glob(f"{sd}/smplx_mesh_betas/*.npy")[0]).reshape(-1, 300)[0].astype(np.float32)
        wav, _ = librosa.load(audio_path, sr=SR)
    except Exception:
        return [(k, None, None, None, None) for k, _ in items]
    out = []
    for k, s in items:
        f0 = round(s * 30)
        m = seq[f0:f0 + W]
        if len(m) < W:
            out.append((k, None, None, None, None)); continue
        yaw = R.from_rotvec(m[0, :3]).apply([0.0, 0.0, 1.0])
        rot = R.from_euler("y", -np.arctan2(yaw[0], yaw[2]))
        pose = m[:, :156].copy()
        pose[:, :3] = (rot * R.from_rotvec(m[:, :3])).as_rotvec()
        tr = rot.apply(m[:, 156:159] - m[0, 156:159] * [1, 0, 1]).astype(np.float32)
        au = wav[round(s * SR):round(s * SR) + A]
        au = np.pad(au, (0, A - len(au)))
        out.append((k, pose, tr, (np.clip(au, -1, 1) * 32767).astype(np.int16), betas))
    return out


@torch.no_grad()
def contacts(model, pose, trans, betas):
    """(n, 300, 156) -> (n, 300, 4) feet contact"""
    n = len(pose)
    p = torch.from_numpy(pose.reshape(n * W, 156)).cuda()
    b = torch.from_numpy(np.repeat(betas, W, 0)).cuda()
    t = torch.from_numpy(trans.reshape(n * W, 3)).cuda()
    js = []
    for s in range(0, n * W, 1200):                   # 4 windows at a time: lbs needs ~1.4 MB per frame
        e = s + 1200
        z = torch.zeros(len(p[s:e]), 3, device="cuda")
        js.append(model(betas=b[s:e], transl=t[s:e], global_orient=p[s:e, :3], body_pose=p[s:e, 3:66],
                        left_hand_pose=p[s:e, 66:111], right_hand_pose=p[s:e, 111:156], jaw_pose=z, leye_pose=z,
                        reye_pose=z, expression=torch.zeros(len(z), 100, device="cuda"),
                        return_joints=True).joints[:, (7, 8, 10, 11)])
    j = torch.cat(js).reshape(n, W, 4, 3)
    v = torch.zeros(n, W, 4, device="cuda")
    v[:, :-1] = (j[:, 1:] - j[:, :-1]).norm(dim=-1)
    return (v < 0.01).cpu().numpy().astype(np.uint8)


def main():
    src, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    model = smplx.create(SMPLX_DIR, model_type="smplx", gender="NEUTRAL_2020", use_face_contour=False, num_betas=300,
                         num_expression_coeffs=100, ext="npz", use_pca=False, batch_size=1).cuda().eval()
    for split in ("val", "train"):
        rows = list(csv.DictReader(open(f"{src}/{split}_windows.csv")))
        N = len(rows)
        by = defaultdict(list)
        for k, r in enumerate(rows):
            by[(os.path.dirname(os.path.dirname(r["audio_filename"])), r["audio_filename"])].append((k, float(r["start_sec"])))
        jobs = [(sd, ap, it) for (sd, ap), it in by.items()]
        mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{out}/{split}_{name}.npy", "w+", dt, shape)
        P, T, C, Au = (mm("pose", (N, W, 156), np.float16), mm("trans", (N, W, 3), np.float32),
                       mm("contact", (N, W, 4), np.uint8), mm("audio", (N, A), np.int16))
        ok = np.zeros(N, bool)
        done = 0
        with ProcessPoolExecutor(32) as ex:
            for res in ex.map(load_rec, jobs, chunksize=4):
                good = [r for r in res if r[1] is not None]
                if good:
                    ks = [r[0] for r in good]
                    pose = np.stack([r[1] for r in good]); tr = np.stack([r[2] for r in good])
                    P[ks] = pose.astype(np.float16); T[ks] = tr; Au[ks] = np.stack([r[3] for r in good])
                    C[ks] = contacts(model, pose, tr, np.stack([r[4] for r in good]))
                    ok[ks] = True
                done += len(res)
                if done % 2000 < len(res):
                    print(f"{split}: {done}/{N}", flush=True)
        for a in (P, T, C, Au):
            a.flush()
        np.save(f"{out}/{split}_ok.npy", ok)
        print(f"{split}: {ok.sum()}/{N} windows ok", flush=True)


if __name__ == "__main__":
    main()
