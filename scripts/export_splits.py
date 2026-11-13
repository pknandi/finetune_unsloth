"""Release file of the Embody-3D evaluation splits (2026-11-07): every validation window with its recording, start time
and evaluation group, built with the exact selection of scripts/aligned_eval.py (group 0 = development windows used to
choose recipes; groups 1-7 = fresh evaluation windows, 60 each, the paper's 420). Train / val recordings are listed too.
usage: python scripts/export_splits.py [data=outputs/stream-hubert-nov04] [out=release]
"""
import os, sys, csv, random
import numpy as np
data = sys.argv[1] if len(sys.argv) > 1 else "outputs/stream-hubert-nov04"
out = sys.argv[2] if len(sys.argv) > 2 else "release"
os.makedirs(out, exist_ok=True)
va = list(csv.DictReader(open(f"{data}/val_windows.csv"))); n = 60
random.seed(0); g0 = sorted(random.sample(range(len(va)), n))
rest = [i for i in np.random.RandomState(0).permutation(len(va)) if i not in set(g0)]
group = {i: 0 for i in g0}
for k in range(1, 8):
    for i in rest[(k - 1) * n:k * n]:
        group[int(i)] = k
rel = lambda a: a.split("/datasets/")[-1] if "/datasets/" in a else a
with open(f"{out}/embody3d_eval_windows.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["val_index", "recording", "start_sec", "duration_sec", "eval_group"])
    for i, r in enumerate(va):
        w.writerow([i, rel(r["audio_filename"]), r["start_sec"], 10.0, group.get(i, "")])
for split in ("train", "val"):
    recs = sorted({rel(r["audio_filename"]) for r in csv.DictReader(open(f"{data}/{split}_windows.csv"))})
    open(f"{out}/embody3d_{split}_recordings.txt", "w").write("\n".join(recs) + "\n")
    print(split, len(recs), "recordings")
print({k: sum(v == k for v in group.values()) for k in range(8)}, "windows per group of", len(va))
