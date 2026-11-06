"""Fix A for the facing problem: rotate each prediction about the world up axis (+Y) so its first
frame faces the same way as the real person's first frame (audio cannot tell which way someone faces
in the room). Writes val<i>_<subject>_<mode>ya.npy next to the originals; score them with
scripts/aligned_eval.py (same args, modes <mode>ya: existing files are scored, not regenerated).

usage: python scripts/align_yaw.py <eval_dir> <data_dir> <mode> [<mode> ...]
"""
import os, sys, re, csv, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from scipy.spatial.transform import Rotation as R
from vqvae_motion_tokenizer import load_smplx_sequence


def heading(rotvec):
    """yaw (rad) of the body's forward axis (+Z) about world +Y"""
    f = R.from_rotvec(rotvec).apply([0.0, 0.0, 1.0])
    return np.arctan2(f[..., 0], f[..., 2])


def set_heading(m, yaw0):
    """rotate (T, 159) SMPL-X motion about +Y so frame 0's heading becomes yaw0"""
    rot = R.from_euler("y", yaw0 - heading(m[0, :3]))
    out = m.copy()
    out[:, :3] = (rot * R.from_rotvec(m[:, :3])).as_rotvec()
    if m.shape[1] >= 159:
        out[:, 156:159] = rot.apply(m[:, 156:159] - m[0, 156:159]) + m[0, 156:159]
    return out


if __name__ == "__main__":
    ev, data, modes = sys.argv[1], sys.argv[2], sys.argv[3:]
    va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
    n = 0
    for mode in modes:
        for p in glob.glob(f"{ev}/val*_{mode}.npy"):
            i = int(re.match(r"val(\d+)_", os.path.basename(p)).group(1))
            audio, start = va[i]["audio_filename"], float(va[i]["start_sec"])
            g0 = load_smplx_sequence(os.path.dirname(os.path.dirname(audio)))[round(start * 30)]
            np.save(p[:-4] + "ya.npy", set_heading(np.load(p), heading(g0[:3])))
            n += 1
    print(f"aligned {n} files")
