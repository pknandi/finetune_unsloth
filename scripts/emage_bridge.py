"""Run EMAGE (CVPR 2024, PantoMatrix code + HF weights) on the same held-out clips as scripts/aligned_eval.py
and save its motion in our format, so aligned_eval / motion_dist_metrics / audio_motion_match score it
like our models. Runs in the EMAGE venv (/data2/home/mubtasim/venvs/emage), not the motion env.

Each 10 s window is turned to face +Z at frame 0 (EMAGE was trained on BEAT2, facing the camera),
generated, and turned back to the real starting heading. Jaw, eyes and face are not in Embody-3D:
they are zero on the way in and dropped on the way out. Foot contact is zero (not computed).
Modes:
  emage       pretrained generator, its own 4-frame seed (the real first 4 frames)
  emagef15    same, the first 15 frames (1 s) given as seed, like our ...f15 modes
  emagevq     the real motion through EMAGE's part VQ-VAEs (upper / hands / lower indices): its tokenizer ceiling

usage: python scripts/emage_bridge.py <data_dir> <out_dir> <n> <modes> [--weights <dir>] [--gen <dir>]
<data_dir> and <n> must match the aligned_eval call (clip choice: random.seed(0), random.sample). Writes
<out_dir>/{val<i>,train0}_<subject>_<mode>.npy, (300, 159) axis-angle: orient, body, left hand, right hand, transl.
"""
import os, sys, csv, glob, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("n", type=int); ap.add_argument("modes")
ap.add_argument("--code", default="/data2/home/mubtasim/ext/PantoMatrix")
ap.add_argument("--weights", default="/data2/home/mubtasim/ext/emage_work/hf/emage_audio")
ap.add_argument("--gen", default="", help="generator dir (default: the pretrained one in --weights)")
args = ap.parse_args()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, args.code)
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
import emage_utils.rotation_conversions as rc

PARTS = ["smplx_mesh_global_orient", "smplx_mesh_body_pose", "smplx_mesh_left_hand_pose",
         "smplx_mesh_right_hand_pose", "smplx_mesh_transl"]
W = 300
dev = torch.device("cuda")


def load_seq(subj_dir):
    """(T, 159) like vqvae_motion_tokenizer.load_smplx_sequence"""
    a = [np.load(glob.glob(f"{subj_dir}/{p}/*.npy")[0]).reshape(-1, d)
         for p, d in zip(PARTS, (3, 63, 45, 45, 3))]
    T = min(len(x) for x in a)
    return np.concatenate([x[:T] for x in a], 1).astype(np.float32)


def heading(rotvec):
    f = R.from_rotvec(rotvec).apply([0.0, 0.0, 1.0])
    return np.arctan2(f[..., 0], f[..., 2])


def turn(m, yaw, origin):
    """rotate (T, 159) motion by yaw about +Y around origin (x, z of the transl)"""
    rot = R.from_euler("y", yaw)
    out = m.copy()
    out[:, :3] = (rot * R.from_rotvec(m[:, :3])).as_rotvec()
    out[:, 156:159] = rot.apply(m[:, 156:159] - origin) + origin
    return out


def to_emage(m):
    """(T, 159) -> (T, 337): 55-joint 6D (jaw/eyes zero), transl, zero foot contact"""
    T = len(m)
    aa = np.zeros((T, 55, 3), np.float32)
    aa[:, 0] = m[:, :3]; aa[:, 1:22] = m[:, 3:66].reshape(T, 21, 3); aa[:, 25:55] = m[:, 66:156].reshape(T, 30, 3)
    r6 = rc.axis_angle_to_rotation_6d(torch.from_numpy(aa)).reshape(T, 330)
    return torch.cat([r6, torch.from_numpy(m[:, 156:159]), torch.zeros(T, 4)], 1)


def from_emage(aa, trans):
    """(T, 165) axis-angle + (T, 3) -> (T, 159)"""
    T = len(aa)
    aa = aa.reshape(T, 55, 3)
    return np.concatenate([aa[:, 0], aa[:, 1:22].reshape(T, 63), aa[:, 25:55].reshape(T, 90), trans], 1).astype(np.float32)


wd = args.weights
vq = EmageVQModel(
    face_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/face"),
    upper_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/upper"),
    lower_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/lower"),
    hands_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/hands"),
    global_model=EmageVAEConv.from_pretrained(f"{wd}/emage_vq/global")).to(dev).eval()
modes = args.modes.split(",")
gen = EmageAudioModel.from_pretrained(args.gen or wd).to(dev).eval() if set(modes) - {"emagevq"} else None


@torch.no_grad()
def run(mode, m, audio):
    """m: (W, 159) canonical motion, audio: (W/30 * 16000,) -> (W, 165) axis-angle, (W, 3) trans"""
    x = to_emage(m)[None].to(dev)
    trans = x[:, :, 330:333]
    if mode == "emagevq":
        idx = vq.map2index(x[:, :, :330], torch.zeros(1, W, 100, device=dev), tar_contact=x[:, :, 333:], tar_trans=trans)
        d = vq.decode(upper_index=idx["upper"], hands_index=idx["hands"], lower_index=idx["lower"],
                      face_index=idx["face"], get_global_motion=True, ref_trans=trans[:, 0])
    else:
        seed = 15 if mode == "emagef15" else gen.cfg.seed_frames
        mask = None
        if seed > gen.cfg.seed_frames:
            mask = torch.ones_like(x); mask[:, :seed] = 0
        a = torch.from_numpy(audio).float()[None].to(dev)
        lat = gen.inference(a, torch.zeros(1, 1, dtype=torch.long, device=dev), vq, masked_motion=x[:, :seed], mask=mask)
        c = gen.cfg
        idx = {k: lat[f"cls_{k}"].argmax(-1) if getattr(c, f"c{k[0]}") > 0 else None for k in ("upper", "hands", "lower")}
        lt = {k: lat[f"rec_{k}"] if getattr(c, f"l{k[0]}") > 0 and getattr(c, f"c{k[0]}") == 0 else None
              for k in ("face", "upper", "hands", "lower")}
        d = vq.decode(face_latent=lt["face"], upper_latent=lt["upper"], hands_latent=lt["hands"], lower_latent=lt["lower"],
                      upper_index=idx["upper"], hands_index=idx["hands"], lower_index=idx["lower"],
                      get_global_motion=True, ref_trans=trans[:, 0])
    aa = d["motion_axis_angle"][0].cpu().numpy()
    tr = d["trans"][0].cpu().numpy()
    n = min(len(aa), W)
    aa, tr = aa[:n], tr[:n]
    if n < W:                                             # inference drops a remainder shorter than the seed
        aa = np.concatenate([aa, np.repeat(aa[-1:], W - n, 0)]); tr = np.concatenate([tr, np.repeat(tr[-1:], W - n, 0)])
    return aa, tr


va_w = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))
tr_w = list(csv.DictReader(open(f"{args.data}/train_windows.csv")))
random.seed(0)
clips = [("val", i, va_w[i]) for i in sorted(random.sample(range(len(va_w)), min(args.n, len(va_w))))]
clips.append(("train", 0, tr_w[0]))
os.makedirs(args.out, exist_ok=True)
for split, i, w in clips:
    audio_path, start = w["audio_filename"], float(w["start_sec"])
    subj_dir = os.path.dirname(os.path.dirname(audio_path))
    f0 = round(start * 30)
    g = load_seq(subj_dir)[f0:f0 + W]
    yaw0, origin = heading(g[0, :3]), g[0, 156:159].copy()
    canon = turn(g, -yaw0, origin)
    canon[:, 156:159] -= origin * np.array([1, 0, 1], np.float32)     # x, z start at 0 (y kept)
    audio, _ = librosa.load(audio_path, sr=16000, offset=start, duration=W / 30)
    audio = np.pad(audio, (0, max(0, W * 16000 // 30 - len(audio))))
    for mode in modes:
        npy = f"{args.out}/{split}{i}_{os.path.basename(subj_dir)}_{mode}.npy"
        if os.path.exists(npy):
            continue
        aa, tr = run(mode, canon, audio)
        tr = tr - tr[:1] + canon[:1, 156:159]                         # EMAGE's y is absolute (BEAT2); keep ours
        pred = from_emage(aa, tr)
        pred[:, 156:159] += origin * np.array([1, 0, 1], np.float32)
        np.save(npy, turn(pred, yaw0, origin))
        print(f"generated {split}{i} {mode}", flush=True)
print("done", flush=True)
