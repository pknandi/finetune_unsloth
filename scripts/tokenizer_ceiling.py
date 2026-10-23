"""Tokenizer ceiling: GT 10 s window -> encode -> decode -> score with evaluate_motion.py
(root-relative MPJPE, root held fixed exactly like inference). The best any LLM can do.

usage: python scripts/tokenizer_ceiling.py <tokenizer_dir> <run_dir> [n_random_rows=30] [out_dir] [sigmas=0]
  tokenizer_dir: holds tokenizer.pt + normalizer.npz
  run_dir:       holds tokenizer_dataset_mapping.csv + audio_windows.csv
Row 0 (BWW760, forced into every overfit run) is always scored, plus n random rows (seed 0,
so every tokenizer is scored on the same rows). Last line: CEILING bww760=<mm> mean=<mm> median=<mm> ...
sigmas: comma list of post-decode smoothing strengths (smooth_features, frames), e.g. 0,1,2,3.
One CEILING line per sigma, in the given order, each with sigma= and jerk= (median jerk ratio vs GT).
"""
import os, sys, json, random, subprocess, csv, tempfile
from concurrent.futures import ThreadPoolExecutor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from vqvae_motion_tokenizer import (VQVAETokenizer, Normalizer, features_to_smplx, preprocess_motion,
                                    load_smplx_sequence, load_audio_windows, smooth_features)

tok_dir, run_dir = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
out = sys.argv[4] if len(sys.argv) > 4 else tempfile.mkdtemp(prefix="ceiling_")
sigmas = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0]
os.makedirs(out, exist_ok=True)

win = load_audio_windows(f"{run_dir}/audio_windows.csv")
rows = [r for r in csv.DictReader(open(f"{run_dir}/tokenizer_dataset_mapping.csv"))
        if os.path.normpath(r["audio_filename"]) in win]
random.seed(0)
b = next((i for i, r in enumerate(rows) if "ACTING_Adult_Birthday_--103301-106600/BWW760" in r["motion_dirname"]), 0)
pick = [b] + random.sample([i for i in range(len(rows)) if i != b], n)   # same picks as before when b == 0
tok = VQVAETokenizer(); tok.load(f"{tok_dir}/tokenizer.pt")
norm = Normalizer(); norm.load(f"{tok_dir}/normalizer.npz")

def score(npy, js, d, start):
    subprocess.run([sys.executable, "evaluate_motion.py", "--session_dir", os.path.dirname(d),
                    "--subject", os.path.basename(d), "--pred_npy", npy,
                    "--smplx_model_dir", "./outputs/smplx/models", "--fps", "30", "--max_seconds", "10.0",
                    "--gt_start_sec", str(start), "--output_json", js], capture_output=True)
    try:
        rr = json.load(open(js))["root_relative"]
        return rr["mpjpe_mm"], rr["dynamics"]["jerk_ratio_pred_over_gt"]
    except Exception:
        return float("nan"), float("nan")

jobs = []   # (sigma, row, npy, js, motion_dir, start)
for i in pick:
    r = rows[i]; start = win[os.path.normpath(r["audio_filename"])]
    off = round(start * 30)
    m = load_smplx_sequence(r["motion_dirname"])[off: off + 300]
    x = norm.transform(preprocess_motion(m, norm.orient_repr, norm.pose_repr))
    feat = tok.decode(tok.encode(x)) * norm.std + norm.mean
    for sg in sigmas:
        s = features_to_smplx(smooth_features(feat, sg), norm.orient_repr, norm.pose_repr)[:, :159]
        s[:, 156:159] = 0
        tag = f"row{i}" if sg == 0 else f"row{i}_s{sg:g}"
        np.save(f"{out}/{tag}.npy", s)
        jobs.append((sg, i, f"{out}/{tag}.npy", f"{out}/{tag}.json", os.path.normpath(r["motion_dirname"]), start))

with ThreadPoolExecutor(6) as ex:
    scores = list(ex.map(lambda j: score(*j[2:]), jobs))
for sg in sigmas:
    mm = [sc[0] for j, sc in zip(jobs, scores) if j[0] == sg]
    jk = [sc[1] for j, sc in zip(jobs, scores) if j[0] == sg]
    for (j, sc) in zip(jobs, scores):
        if j[0] == sg:
            print(f"sigma {sg:g} row {j[1]:4d} {os.path.basename(j[4])} start {j[5]:5.1f}  ceiling {sc[0]:6.1f} mm  jerk {sc[1]:5.2f}x", flush=True)
    print(f"CEILING bww760={mm[0]:.1f} mean={np.nanmean(mm):.1f} median={np.nanmedian(mm):.1f} rows={len(mm)} "
          f"sigma={sg:g} jerk={np.nanmedian(jk):.2f} bww760_jerk={jk[0]:.2f}", flush=True)
