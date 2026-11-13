"""BEAT2 jerk ratio (2026-11-12): per test recording, SMPL-X joint positions (body + hands, as scripts/beat2_score.py,
GT betas, jaw / eyes from GT) of the prediction and the real motion; jerk = mean norm of the third difference of the
positions (30 fps); ratio = prediction / real; prints the median and mean over recordings (1 = as smooth as real).
usage: python scripts/beat2_jerk.py <beat2_dir> <recordings.csv> <pred_dir>[,<pred_dir2>...]
"""
import os, sys, csv
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
beat2, recs, preds = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), [os.path.abspath(p) for p in sys.argv[3].split(",")]
code, tools = "/data2/home/mubtasim/ext/PantoMatrix", "/data2/home/mubtasim/ext/emage_work/hf/"
sys.path.insert(0, code); sys.path.insert(0, tools); os.chdir(tools)
from emage_utils.motion_rep_transfer import get_motion_rep_numpy

ids = [r["id"] for r in csv.DictReader(open(recs)) if r["split"] == "test"]
jerk = lambda p: np.linalg.norm(np.diff(p, 3, axis=0), axis=-1).mean()
for pd in preds:
  rs = []
  try:
    for rid in ids:
        g = np.load(f"{beat2}/smplxflame_30/{rid}.npz", allow_pickle=True)
        gt, betas = g["poses"], g["betas"]
        pred = np.load(f"{pd}/{rid}_output.npz", allow_pickle=True)["poses"]
        t = min(len(gt), len(pred)); gt, pred = gt[:t].astype(np.float32), pred[:t].astype(np.float32).copy()
        pred[:, 66:75] = gt[:, 66:75]
        P = lambda x: get_motion_rep_numpy(x, device="cuda", betas=betas)["position"].reshape(t, -1, 3)[:, list(range(22)) + list(range(25, 55))]
        rs.append(jerk(P(pred)) / jerk(P(gt)))
  except Exception as ex:
    print(f"B2JERK pred={pd} FAILED {type(ex).__name__}: {ex}", flush=True); continue
  if rs:
    print(f"B2JERK pred={pd} median={np.median(rs):.2f} mean={np.mean(rs):.2f} n={len(rs)}", flush=True)
