"""BEAT2 benchmark scores with EMAGE's own tools (emage_evaltools: FGD, BC, L1div), following PantoMatrix
train_emage_audio.evaluation_fn: per test recording, cut to the shorter of GT / prediction; BC and L1div on
SMPL-X joint positions (body + hands, root and jaw zeroed, GT betas), BC skipping the first and last 2 s;
FGD on 55-joint 6D rotations with EMAGE's AESKConv encoder. Runs in the EMAGE venv.
--jaw gt puts the real jaw / eye rotations into the prediction (for models that, like ours, do not
generate them; they only enter FGD). Prediction files: <pred_dir>/<id>_output.npz with poses (T,165).
Prints BEAT2 pred=<dir> fgd= bc= l1div= n=   (pred_dir "gt" scores the real motion against itself).

--gt_parts root,lower,... puts those joint groups back to GT (which part costs FGD).
usage: python scripts/beat2_score.py <beat2_dir> <recordings.csv> <pred_dir|gt> [--jaw gt|keep] [--gt_parts ...]
"""
import os, sys, csv, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("beat2"); ap.add_argument("recs"); ap.add_argument("pred")
ap.add_argument("--jaw", default="keep", choices=["keep", "gt"])
ap.add_argument("--split", default="test")
ap.add_argument("--gt_parts", default="", help="comma list of root,lower,upper,hands put back to GT (FGD-gap analysis)")
ap.add_argument("--code", default="/data2/home/mubtasim/ext/PantoMatrix")
ap.add_argument("--tools", default="/data2/home/mubtasim/ext/emage_work/hf/")
args = ap.parse_args()
sys.path.insert(0, args.code); sys.path.insert(0, args.tools)
os.chdir(args.tools)     # emage_utils.motion_rep_transfer loads ./emage_evaltools/smplx_models
from emage_evaltools.mertic import FGD, BC, L1div
from emage_utils.motion_rep_transfer import get_motion_rep_numpy
import emage_utils.rotation_conversions as rc

dev = "cuda"
PARTS = {"root": [0], "lower": [1, 2, 4, 5, 7, 8, 10, 11], "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
         "hands": list(range(25, 55))}     # SMPL-X joint ids (22-24 jaw / eyes: --jaw)
fgd = FGD(download_path="./emage_evaltools/")
bc = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
l1 = L1div()
ids = [r["id"] for r in csv.DictReader(open(args.recs)) if r["split"] == args.split]
for rid in ids:
    g = np.load(f"{args.beat2}/smplxflame_30/{rid}.npz", allow_pickle=True)
    gt, betas = g["poses"], g["betas"]
    pred = gt if args.pred == "gt" else np.load(f"{args.pred}/{rid}_output.npz", allow_pickle=True)["poses"]
    t = min(len(gt), len(pred))
    gt, pred = gt[:t].astype(np.float32), pred[:t].astype(np.float32).copy()
    if args.jaw == "gt":
        pred[:, 66:75] = gt[:, 66:75]
    for part in filter(None, args.gt_parts.split(",")):
        for j in PARTS[part]:
            pred[:, 3 * j:3 * j + 3] = gt[:, 3 * j:3 * j + 3]
    pos = get_motion_rep_numpy(pred, device=dev, betas=betas)["position"].reshape(t, -1)
    a_beat = bc.load_audio(f"{args.beat2}/wave16k/{rid}.wav", t_start=2 * 16000, t_end=int((t - 60) / 30 * 16000))
    m_beat = bc.load_motion(pos, t_start=60, t_end=t - 60, pose_fps=30, without_file=True)
    bc.compute(a_beat, m_beat, length=t - 120, pose_fps=30)
    l1.compute(pos)
    to6 = lambda x: rc.axis_angle_to_rotation_6d(torch.from_numpy(x).to(dev).reshape(1, t, 55, 3)).reshape(1, t, 330)
    fgd.update(to6(pred).float(), to6(gt).float())
print(f"BEAT2 pred={args.pred} jaw={args.jaw} gt_parts={args.gt_parts or '-'} fgd={fgd.compute():.3f} bc={bc.avg():.3f} l1div={l1.avg():.3f} n={len(ids)}",
      flush=True)
