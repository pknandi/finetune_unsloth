"""Our model (speech_to_motion_small.py checkpoint) on whole BEAT2 test recordings, for scripts/beat2_score.py.
The model makes 10 s windows (150 tokens at 15 fps); a recording is covered by windows every 9 s, each one
seeded with the last 1 s of the previous window's output: that 1 s is decoded, re-encoded in the new
window's frame (6d_rel: facing +Z and translation 0 at the window start) and forced as the first 15 tokens,
then the window is turned back to the previous output's heading / position at its start. Window 0 is
seeded with the real first --seed_frames tokens (0 = no seed, real starting heading only).
Jaw and eyes are zero (not in our features; beat2_score.py --jaw gt fills them from GT for FGD).
Writes <out>/<id>_output.npz (poses (T,165), trans, betas, expressions, mocap_frame_rate).

usage: python scripts/beat2_generate.py <ckpt_dir> <tok_dir> <beat2_out_dir> <out_dir>
       [--temperature 1.3] [--top_p 1] [--guidance 3.5] [--sigma 2.5] [--seed_frames 15] [--draw 0] [--energy K]
"""
import os, sys, csv, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("tok"); ap.add_argument("b2"); ap.add_argument("out")
ap.add_argument("--temperature", type=float, default=1.3); ap.add_argument("--top_p", type=float, default=1.0)
ap.add_argument("--guidance", type=float, default=3.5); ap.add_argument("--sigma", type=float, default=2.5)
ap.add_argument("--seed_frames", type=int, default=15); ap.add_argument("--draw", type=int, default=0)
ap.add_argument("--split", default="test"); ap.add_argument("--lookahead", type=int, default=8)
ap.add_argument("--ceiling", action="store_true", help="the real tokens of each window decoded: the tokenizer's best")
ap.add_argument("--max_recs", type=int, default=0, help="smoke test: first N recordings")
ap.add_argument("--adapt", type=float, nargs=2, default=None, help="uncertainty-adaptive guidance (P2): g_min g_max")
ap.add_argument("--energy_kind", default="c", help="c: reward-matcher energy; t: privileged teacher energy (--energy_teacher, full-window audio)")
ap.add_argument("--energy_teacher", default="outputs/beat2-ft-f2hw1.0-nov01/best")
ap.add_argument("--energy_thead", default="outputs/energy-teacher-b2-nov06/energy_teacher.pt", help="privileged energy head (kind T)")
ap.add_argument("--energy_head", default="", help="distilled causal energy head (kind d)")
ap.add_argument("--energy", type=int, default=0, help="energy-guided decoding (scripts/energy_decode.py), K candidates per 1 s chunk, causal")
ap.add_argument("--stream", action="store_true", help="online-honest audio: HubertFeaturizer.stream (each slot from audio up to its end)")
args = ap.parse_args()
from speech_to_motion_small import load_trained, generate
from speech_to_motion_aligned import read_window, HubertFeaturizer
from speech_to_motion_inference import motion_ids_to_smplx
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion, heading_yaw

T_M, FPS_M, FPS = 150, 15, 30
DUR = (T_M + args.lookahead) / FPS_M
NF = (int(DUR * 50) - 1) // 2                  # 25 fps HuBERT frames per window, as build_dense.py
SEED = 15                                      # chained seed: 1 s
STEP = T_M - SEED                              # 9 s between window starts (in tokens)
torch.manual_seed(args.draw); np.random.seed(args.draw)
model, io, bos = (None, None, None) if args.ceiling else load_trained(args.ckpt)
norm = Normalizer(); norm.load(f"{args.tok}/normalizer.npz")
tok = VQVAETokenizer(); tok.load(f"{args.tok}/tokenizer.pt")
feat = HubertFeaturizer()
os.makedirs(args.out, exist_ok=True)


def encode(seq):
    """(n, 159) SMPL-X frames (n <= 316) -> the first tokens of a window starting at seq[0]"""
    seq = np.concatenate([seq, np.repeat(seq[-1:], 2 * T_M + 16 - len(seq), 0)])
    return tok.encode(norm.transform(preprocess_motion(seq, norm.orient_repr, norm.pose_repr)))


def place(motion, g0):
    """turn a decoded window (facing +Z, translation 0 at frame 0) to frame g0's heading and position"""
    rot = Rotation.from_euler("y", heading_yaw(g0[:3]))
    out = motion.copy()
    out[:, :3] = (rot * Rotation.from_rotvec(motion[:, :3])).as_rotvec()
    out[:, 156:159] = rot.apply(motion[:, 156:159] - motion[:1, 156:159]) + g0[156:159]
    return out


recs = [r for r in csv.DictReader(open(f"{args.b2}/split/recordings.csv")) if r["split"] == args.split]
recs = recs[:args.max_recs or None]
for r in recs:
    rid = r["id"]
    if os.path.exists(f"{args.out}/{rid}_output.npz"):
        continue
    mdir = f"{args.b2}/data/{rid}/{rid.split('_')[1]}"
    audio = f"{mdir}/audio_separated/{rid}.wav"
    gt = load_smplx_sequence(mdir)
    n = min(len(gt), int(float(r["seconds"]) * FPS))
    out = np.zeros((0, 159), np.float32)
    k = 0
    while len(out) < n:
        s = k * STEP / FPS_M                          # window start, seconds
        w = read_window(audio, s, DUR)
        if args.stream:
            f = feat.stream(w, NF, T_M + args.lookahead)
        else:
            h = feat(w[None])[0, :2 * NF].astype(np.float32)
            f = ((h[0::2] + h[1::2]) / 2).astype(np.float16)
        if k == 0:
            prefix = encode(gt[:2 * T_M + 16])[:args.seed_frames] if args.seed_frames else None
            g0 = gt[0]
        else:
            a = round(s * FPS)
            prefix, g0 = encode(out[a:a + 2 * SEED])[:SEED], out[a]
        if args.ceiling:
            a = round(s * FPS)
            ids = encode(gt[a:a + 2 * T_M + 16])[:T_M]
            g0 = gt[a]
        elif args.energy:
            from energy_decode import generate_energy
            causal = np.repeat(np.asarray(f, np.float32), 2, 0)
            tf = None
            if args.energy_kind in ("t", "T"):              # the offline teacher's full-window features (privileged)
                h = feat(w[None])[0, :2 * NF].astype(np.float32)
                tf = ((h[0::2] + h[1::2]) / 2).astype(np.float16)
            ids = generate_energy(model, f, causal, causal, args.tok, K=args.energy, energy=args.energy_kind,
                                  temperature=args.temperature, guidance=args.guidance, prefix=prefix, sigma=args.sigma,
                                  teacher=args.energy_teacher, t_feats=tf, t_head=args.energy_thead,
                                  head=args.energy_head or None, adapt=tuple(args.adapt) if args.adapt else None)
        else:
            ids = generate(model, io, bos, f, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                           prefix=prefix, guidance=args.guidance, adapt=tuple(args.adapt) if args.adapt else None)
        m = place(motion_ids_to_smplx(ids, f"{args.tok}/tokenizer.pt", f"{args.tok}/normalizer.npz", args.sigma), g0)
        a = round(s * FPS)
        out = np.concatenate([out[:a + (2 * SEED if k else 0)], m[2 * SEED if k else 0:]])
        k += 1
    out = out[:n]
    poses = np.zeros((n, 165), np.float32)
    poses[:, :66], poses[:, 75:165] = out[:, :66], out[:, 66:156]
    betas = np.load(f"{mdir}/smplx_mesh_betas/{rid}.npy").reshape(-1)
    np.savez(f"{args.out}/{rid}_output.npz", poses=poses, trans=out[:, 156:159], betas=betas,
             expressions=np.zeros((n, 100), np.float32), mocap_frame_rate=FPS)
    print(f"{rid}: {n} frames, {k} windows", flush=True)
