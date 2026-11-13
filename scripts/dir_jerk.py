"""Jerk ratio (generated / real, median over windows) for an eval dir written outside scripts/aligned_eval.py
(baselines: EMAGE, Super Star, MIBURI, nearest neighbour). Same scorer as aligned_eval's HELDOUT jerk:
evaluate_motion.py per window (root-relative dynamics, jerk_ratio_pred_over_gt), 4 threads, no video.
usage: python scripts/dir_jerk.py <eval_dir>[,<eval_dir2>...] <mode> [data=outputs/run-full-q8-oct21/aligned_data]
prints: DIRJERK mode=<mode> jerk=<median> n=<windows>
"""
import os, sys, re, csv, glob, json, subprocess
from concurrent.futures import ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import numpy as np

dirs, mode = sys.argv[1].split(","), sys.argv[2]
data = sys.argv[3] if len(sys.argv) > 3 else "outputs/run-full-q8-oct21/aligned_data"
va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
jobs = []
for d in dirs:
    for p in sorted(glob.glob(f"{d}/val*_{mode}.npy")):
        m = re.match(rf"val(\d+)_.*_{re.escape(mode)}\.npy$", os.path.basename(p))
        if m:
            w = va[int(m.group(1))]
            jobs.append((p, w["audio_filename"], float(w["start_sec"])))


def score(job):
    npy, audio, start = job
    subj = os.path.dirname(os.path.dirname(audio)); js = npy[:-4] + ".json"
    if not os.path.exists(js):
        subprocess.run([sys.executable, "evaluate_motion.py", "--session_dir", os.path.dirname(subj),
                        "--subject", os.path.basename(subj), "--pred_npy", npy, "--audio_path", audio,
                        "--smplx_model_dir", "./outputs/smplx/models", "--fps", "30", "--max_seconds", "10.0",
                        "--gt_start_sec", str(start), "--output_json", js], capture_output=True)
    try:
        return json.load(open(js))["root_relative"]["dynamics"]["jerk_ratio_pred_over_gt"]
    except Exception:
        return None


with ThreadPoolExecutor(4) as ex:
    r = [x for x in ex.map(score, jobs) if x is not None]
print(f"DIRJERK mode={mode} dirs={len(dirs)} jerk={np.median(r):.2f} n={len(r)} of {len(jobs)}", flush=True)
