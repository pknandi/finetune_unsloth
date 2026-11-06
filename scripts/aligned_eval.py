"""Evaluate a speech_to_motion_aligned.py or speech_to_motion_small.py checkpoint: generate motion for n held-out val clips (and
training row 0, BWW760 in the acting+polyadic set, as the memorization check), score each against GT
(root-relative MPJPE, jerk ratio, beat consistency) and render comparison videos. Modes:
  greedy | sample (temperature 1, top-p 0.9; sample0.8 = temperature 0.8; samplep1 = top-p 1; sample1.3p1c2 = temperature 1.3
  on RVQ levels 0-1 only; sample1.3p1f15 = first 15 motion frames (1 s) forced to the real tokens) | seedhold15 (the real
  pose at 1 s, held still: the floor for seeded modes) | ceiling (the GT tokens decoded: the tokenizer's best)
  static (the training mean pose held still: the floor a model must beat)
<ckpt_dir> is not loaded (pass any path) when modes are only ceiling/static.

usage: python scripts/aligned_eval.py <ckpt_dir> <data_dir> <tokenizer_dir> <out_dir> [n=10] [modes=greedy,sample,ceiling] [sigma=1.5]
NO_VIDEO=1 skips the comparison videos (much faster). EVAL_GROUP=k (1..7 for 60 of 531) scores a further
disjoint group of n val windows instead of the default one. Last lines: HELDOUT mode=<m> mpjpe= jerk= bc= bc_gt= n=   and   TRAIN0 mode=<m> mpjpe= jerk=
"""
import os, sys, re, csv, json, random, subprocess
from concurrent.futures import ThreadPoolExecutor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
from speech_to_motion_inference import motion_ids_to_smplx
from vqvae_motion_tokenizer import Normalizer, features_to_smplx

ckpt, data, tok_dir, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 10
modes = sys.argv[6].split(",") if len(sys.argv) > 6 else ["greedy", "sample", "ceiling"]
sigma = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5
os.makedirs(out, exist_ok=True)

va_w = list(csv.DictReader(open(f"{data}/val_windows.csv")))
tr_w = list(csv.DictReader(open(f"{data}/train_windows.csv")))
va_f = np.load(f"{data}/val_hubert.npy", mmap_mode="r"); va_m = np.load(f"{data}/val_motion.npy")
tr_f = np.load(f"{data}/train_hubert.npy", mmap_mode="r"); tr_m = np.load(f"{data}/train_motion.npy")
if os.path.exists(f"{ckpt}/small.pt") and torch.load(f"{ckpt}/small.pt", map_location="cpu")["cfg"].get("partner"):
    from speech_to_motion_small import Join                 # conversational model: append the partner features
    va_f = Join(va_f, np.load(f"{data}/val_partner.npy", mmap_mode="r"))
    tr_f = Join(tr_f, np.load(f"{data}/train_partner.npy", mmap_mode="r") if os.path.exists(f"{data}/train_partner.npy") else None)
random.seed(0)
pick = sorted(random.sample(range(len(va_w)), min(n, len(va_w))))
if int(os.environ.get("EVAL_GROUP", "0")):
    # EVAL_GROUP=k >= 1: the k-th further group of n val windows, disjoint from the default group and from each other
    rest = [i for i in np.random.RandomState(0).permutation(len(va_w)) if i not in set(pick)]
    k = int(os.environ["EVAL_GROUP"]); pick = sorted(rest[(k - 1) * n:k * n])
    assert len(pick) == n, f"not enough val windows for group {k}"
clips = [("val", i, va_w[i], va_f[i], va_m[i]) for i in pick]
clips.append(("train", 0, tr_w[0], tr_f[0], tr_m[0]))
lm = None
if set(modes) - {"ceiling", "static"}:
    # a speech_to_motion_small.py checkpoint has small.pt; otherwise it is an aligned LLM checkpoint
    if os.path.exists(f"{ckpt}/masked.pt"):
        from speech_to_motion_masked import load_trained, generate
    elif os.path.exists(f"{ckpt}/small.pt"):
        from speech_to_motion_small import load_trained, generate
    else:
        from speech_to_motion_aligned import load_trained, generate
    lm, io, bos = load_trained(ckpt)
norm = Normalizer(); norm.load(f"{tok_dir}/normalizer.npz")
static = features_to_smplx(np.repeat(np.asarray(norm.mean).reshape(1, -1), 300, 0), norm.orient_repr, norm.pose_repr)[:, :159]
static[:, 156:] = 0
REL = norm.orient_repr == "6d_rel"


def place(motion, audio, start):
    """6d_rel tokenizers decode facing +Z at frame 0 (the window's starting heading is removed in
    training): turn the output to the real person's starting heading, as a seed pose would"""
    if not REL:
        return motion
    from scipy.spatial.transform import Rotation
    from vqvae_motion_tokenizer import load_smplx_sequence, heading_yaw
    g0 = load_smplx_sequence(os.path.dirname(os.path.dirname(audio)))[round(start * 30)]
    rot = Rotation.from_euler("y", heading_yaw(g0[:3]))
    out = motion.copy()
    out[:, :3] = (rot * Rotation.from_rotvec(motion[:, :3])).as_rotvec()
    out[:, 156:159] = rot.apply(motion[:, 156:159])
    return out

jobs = []   # (split, mode, tag, npy, subj_dir, audio, start)
for split, i, w, f, m in clips:
    audio, start = w["audio_filename"], float(w["start_sec"])
    subj_dir = os.path.dirname(os.path.dirname(audio))     # .../<session>/<subject>
    for mode in modes:
        tag = f"{split}{i}_{os.path.basename(subj_dir)}_{mode}"
        npy = f"{out}/{tag}.npy"
        if not os.path.exists(npy):
            if mode == "static":
                np.save(npy, place(static, audio, start))
                jobs.append((split, mode, tag, npy, subj_dir, audio, start))
                continue
            if mode.startswith("seedhold"):
                # the real pose at the end of the seed, held still: the floor for seeded (...f<n>) modes
                nf = int(mode[8:] or 15)
                g = motion_ids_to_smplx(m[(m >= 0).all(1)].astype(np.int64), f"{tok_dir}/tokenizer.pt", f"{tok_dir}/normalizer.npz", sigma)
                k = min(len(g) - 1, nf * len(g) // len(m))
                np.save(npy, place(np.concatenate([g[:k], np.repeat(g[k:k + 1], len(g) - k, 0)]), audio, start))
                jobs.append((split, mode, tag, npy, subj_dir, audio, start))
                continue
            if mode == "ceiling":
                ids = m[(m >= 0).all(1)].astype(np.int64)
            else:
                # sample[<temperature>][p<top_p>], e.g. sample0.8, samplep1 (temperature 1, top-p 1)
                # ...c<n>: the temperature only on the first n RVQ levels (small model only)
                # ...g<s>: audio guidance scale s (small model trained with --cond_drop)
                # ...f<n>: the first n motion frames forced to the real tokens (seed motion, small model only)
                # ...s<k>: another independent sample with the same settings (repeat draws for error bars)
                # ...a<gmin>-<gmax>: guidance per frame from g_min to g_max by the model's uncertainty (P2)
                # ...e<K><c|o<d>>: energy-guided decoding (scripts/energy_decode.py): best of K candidates per 1 s chunk by
                # the reward matcher's energy, causal (c) or oracle with d 50 fps frames of future audio (o<d>)
                mt = re.match(r"sample([0-9.]*)(?:p([0-9.]+))?(?:c(\d+))?(?:f(\d+))?(?:g([0-9.]+))?(?:e(\d+)(c|o\d+))?(?:a([0-9.]+)-([0-9.]+))?(?:s(\d+))?$", mode)
                kw = {"temperature": float(mt.group(1) or 1)} if mt else {}
                if mt and mt.group(2):
                    kw["top_p"] = float(mt.group(2))
                if mt and mt.group(3):
                    kw["temp_levels"] = int(mt.group(3))
                if mt and mt.group(4):
                    kw["prefix"] = m[:int(mt.group(4))].astype(np.int64)
                if mt and mt.group(5):
                    kw["guidance"] = float(mt.group(5))
                if mt and mt.group(8):                                 # ...a<gmin>-<gmax>: uncertainty-adaptive guidance (P2)
                    kw["adapt"] = (float(mt.group(8)), float(mt.group(9)))
                if mt and mt.group(6) and split == "val":
                    sys.path.insert(0, f"{REPO}/scripts"); from energy_decode import generate_energy
                    full = np.asarray(np.load("outputs/run-full-q8-oct21/aligned_data/val_hubert.npy", mmap_mode="r")[i], np.float32)
                    full = np.repeat((full[0:len(full) // 2 * 2:2] + full[1:len(full) // 2 * 2:2]) / 2, 2, 0)   # as the reward matcher was trained
                    causal = np.repeat(np.asarray(f, np.float32)[:, :1024], 2, 0)
                    kw.pop("top_p", None); kw.pop("temp_levels", None)
                    ids = generate_energy(lm, f, causal, full, tok_dir, K=int(mt.group(6)), energy=mt.group(7), sigma=sigma, **kw)
                else:
                    ids = generate(lm, io, bos, f, do_sample=bool(mt), **kw)
            np.save(f"{out}/{tag}_ids.npy", ids)
            np.save(npy, place(motion_ids_to_smplx(ids, f"{tok_dir}/tokenizer.pt", f"{tok_dir}/normalizer.npz", sigma), audio, start))
            print(f"generated {tag}", flush=True)
        jobs.append((split, mode, tag, npy, subj_dir, audio, start))
del lm


def score(job):
    split, mode, tag, npy, subj_dir, audio, start = job
    common = ["--session_dir", os.path.dirname(subj_dir), "--subject", os.path.basename(subj_dir),
              "--pred_npy", npy, "--audio_path", audio, "--smplx_model_dir", "./outputs/smplx/models",
              "--fps", "30", "--max_seconds", "10.0", "--gt_start_sec", str(start)]
    js = f"{out}/{tag}.json"
    subprocess.run([sys.executable, "evaluate_motion.py", *common, "--output_json", js], capture_output=True)
    if not os.environ.get("NO_VIDEO"):
        subprocess.run([sys.executable, "compare_motion.py", *common, "--output_path", f"{out}/{tag}.mp4"],
                       capture_output=True)
    try:
        d = json.load(open(js))
        return (d["root_relative"]["mpjpe_mm"], d["root_relative"]["dynamics"]["jerk_ratio_pred_over_gt"],
                d["beat_consistency"]["bc_pred"], d["beat_consistency"]["bc_gt"])
    except Exception:
        return None


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(score, jobs))
for job, r in zip(jobs, res):
    print(f"{job[2]:50s} " + ("FAILED" if r is None else
          f"mpjpe {r[0]:6.1f} mm  jerk {r[1]:5.2f}x  bc {r[2]:.3f} (gt {r[3]:.3f})"), flush=True)
for mode in modes:
    rs = [r for job, r in zip(jobs, res) if job[0] == "val" and job[1] == mode and r is not None]
    if rs:
        a = np.array(rs, dtype=float)
        print(f"HELDOUT mode={mode} mpjpe={a[:, 0].mean():.1f} jerk={np.median(a[:, 1]):.2f} "
              f"bc={np.nanmean(a[:, 2]):.3f} bc_gt={np.nanmean(a[:, 3]):.3f} n={len(rs)}", flush=True)
    t = [r for job, r in zip(jobs, res) if job[0] == "train" and job[1] == mode and r is not None]
    if t:
        print(f"TRAIN0 mode={mode} mpjpe={t[0][0]:.1f} jerk={t[0][1]:.2f}", flush=True)
