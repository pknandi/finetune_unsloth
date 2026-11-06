"""BEAT2 (EMAGE, CVPR 2024; huggingface H-Liu1997/BEAT2) -> the Embody-3D folder layout our loaders read
(vqvae_motion_tokenizer.load_smplx_sequence, build_dense.py, aligned_eval.py), plus window lists for the
standard benchmark split (train_test_split.csv; EMAGE / LOM / Super Star train and test on speaker 2 only).

Layout written:  <out>/data/<id>/<speaker>/smplx_mesh_{global_orient,body_pose,left_hand_pose,right_hand_pose,
transl,betas}/<id>.npy   and   <out>/data/<id>/<speaker>/audio_separated/<id>.wav (symlink to wave16k).
BEAT2 poses are 55 joints x 3 axis-angle (root, 21 body, jaw, 2 eyes, 15 + 15 hands); jaw and eyes are
dropped (not in our features), betas cut to 10. Both datasets are y-up at 30 fps.
Window lists in <out>/split/: {train,val,test}_windows.csv (audio_filename, start_sec; every --stride s
window that fits) and recordings.csv (id, split, seconds).

usage: python scripts/beat2_convert.py <beat_english_v2.0.0 dir> <out_dir> [--speakers 2] [--stride 10]
"""
import os, sys, csv, argparse
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--speakers", default="2", help="comma list of speaker ids, or 'all'")
ap.add_argument("--stride", type=float, default=10.0)
args = ap.parse_args()
DUR, FPS = 10.0, 30
spk = None if args.speakers == "all" else set(args.speakers.split(","))

rows = list(csv.DictReader(open(f"{args.src}/train_test_split.csv")))
idk, typek = list(rows[0])[:2]
os.makedirs(f"{args.out}/split", exist_ok=True)
wins = {"train": [], "val": [], "test": []}
recs = []
for r in rows:
    rid, split = r[idk], r[typek]
    if spk is not None and rid.split("_")[0] not in spk:
        continue
    if split not in wins:     # BEAT2 also lists 'additional' (extra train takes in EMAGE)
        split = "train" if split == "additional" else None
    npz, wav = f"{args.src}/smplxflame_30/{rid}.npz", f"{args.src}/wave16k/{rid}.wav"
    if split is None or not (os.path.exists(npz) and os.path.exists(wav)):
        continue
    d = np.load(npz, allow_pickle=True)
    assert int(d["mocap_frame_rate"]) == FPS, (rid, d["mocap_frame_rate"])
    p = d["poses"].astype(np.float32)
    rdir = f"{args.out}/data/{rid}/{rid.split('_')[1]}"
    parts = {"global_orient": p[:, :3], "body_pose": p[:, 3:66], "left_hand_pose": p[:, 75:120],
             "right_hand_pose": p[:, 120:165], "transl": d["trans"].astype(np.float32),
             "betas": d["betas"][:10].astype(np.float32)[None]}
    for k, v in parts.items():
        os.makedirs(f"{rdir}/smplx_mesh_{k}", exist_ok=True)
        np.save(f"{rdir}/smplx_mesh_{k}/{rid}.npy", v)
    os.makedirs(f"{rdir}/audio_separated", exist_ok=True)
    a = f"{rdir}/audio_separated/{rid}.wav"
    if not os.path.lexists(a):
        os.symlink(os.path.abspath(wav), a)
    sec = min(len(p) / FPS, sf.info(wav).duration)
    recs.append((rid, split, round(sec, 2)))
    s = 0.0
    while s + DUR + 1.0 <= sec:     # + 1 s: the model's audio lookahead (8 frames at 15 fps = 0.53 s)
        wins[split].append((a, s)); s += args.stride
for split, w in wins.items():
    with open(f"{args.out}/split/{split}_windows.csv", "w", newline="") as f:
        csv.writer(f).writerows([("audio_filename", "start_sec")] + w)
with open(f"{args.out}/split/recordings.csv", "w", newline="") as f:
    csv.writer(f).writerows([("id", "split", "seconds")] + recs)
for split in wins:
    n = [x for x in recs if x[1] == split]
    print(f"{split}: {len(n)} recordings, {sum(x[2] for x in n) / 3600:.2f} h, {len(wins[split])} windows", flush=True)
