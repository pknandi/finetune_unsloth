"""Held-out check: generate motion for unseen val clips under several decoding modes, score each
against its GT (root-relative MPJPE, jerk ratio, beat consistency) and render comparison videos.
Loads the 8B model once and reuses it for every clip and mode.

usage: python scripts/heldout_eval.py <run_dir> <lora_ckpt> <num_quantizers> <out_dir>
                                      [n_clips=5] [modes=greedy,sample,sample_l0] [smooth_sigma=1.5]
  modes: greedy | sample (all RVQ levels sampled) | sample_l0 (level 0 sampled, finer levels greedy)
Clips: n random rows (seed 0) of <run_dir>/speech_motion_val_audio_windows.csv.
Last lines: HELDOUT mode=<m> mpjpe=<mean mm> jerk=<median ratio> bc=<mean pred BC> bc_gt=<mean GT BC> n=<clips>
"""
import os, sys, csv, json, random, subprocess
from concurrent.futures import ThreadPoolExecutor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from speech_to_motion_inference import load_lm, generate_motion_from_audio

run_dir, ckpt, Q, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 5
modes = sys.argv[6].split(",") if len(sys.argv) > 6 else ["greedy", "sample", "sample_l0"]
sigma = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5
os.makedirs(out, exist_ok=True)

rows = list(csv.DictReader(open(f"{run_dir}/speech_motion_val_audio_windows.csv")))
random.seed(0)
clips = random.sample(rows, min(n, len(rows)))
tok_dir = f"{run_dir}/motion_tokenizer_artifacts"
lm = load_lm("unsloth/llama-3-8b-bnb-4bit", ckpt, Q)

jobs = []   # (mode, tag, npy, subject_dir, audio, start)
for k, r in enumerate(clips):
    audio, start = r["audio_filename"], float(r["start_sec"])
    subj_dir = os.path.dirname(os.path.dirname(audio))     # .../<session>/<subject>
    for mode in modes:
        tag = f"clip{k}_{os.path.basename(subj_dir)}_{mode}"
        npy = f"{out}/{tag}.npy"
        if not os.path.exists(npy):
            generate_motion_from_audio(
                audio, ckpt, f"{tok_dir}/tokenizer.pt", f"{tok_dir}/normalizer.npz", npy,
                "unsloth/llama-3-8b-bnb-4bit", do_sample=(mode != "greedy"),
                sample_levels=(1 if mode == "sample_l0" else None), num_quantizers=Q,
                start_sec=start, smooth_sigma=sigma, lm=lm, ids_npy_path=f"{out}/{tag}_ids.npy")
        jobs.append((mode, tag, npy, subj_dir, audio, start))
del lm

def score(job):
    mode, tag, npy, subj_dir, audio, start = job
    if not os.path.exists(npy):
        return None
    common = ["--session_dir", os.path.dirname(subj_dir), "--subject", os.path.basename(subj_dir),
              "--pred_npy", npy, "--audio_path", audio, "--smplx_model_dir", "./outputs/smplx/models",
              "--fps", "30", "--max_seconds", "10.0", "--gt_start_sec", str(start)]
    js = f"{out}/{tag}.json"
    subprocess.run([sys.executable, "evaluate_motion.py", *common, "--output_json", js], capture_output=True)
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
    print(f"{job[1]:60s} " + ("FAILED" if r is None else
          f"mpjpe {r[0]:6.1f} mm  jerk {r[1]:5.2f}x  bc {r[2]:.3f} (gt {r[3]:.3f})"), flush=True)
for mode in modes:
    rs = [r for job, r in zip(jobs, res) if job[0] == mode and r is not None]
    if not rs:
        print(f"HELDOUT mode={mode} FAILED"); continue
    a = np.array(rs)
    print(f"HELDOUT mode={mode} mpjpe={a[:, 0].mean():.1f} jerk={np.median(a[:, 1]):.2f} "
          f"bc={a[:, 2].mean():.3f} bc_gt={a[:, 3].mean():.3f} n={len(rs)}", flush=True)
