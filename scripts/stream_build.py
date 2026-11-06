"""A dense data dir (scripts/build_dense.py layout) with online-honest audio features (HubertFeaturizer.stream: each
motion slot from the audio up to its own end), for training the strict online model on what it would see live.
Full streaming costs ~0.8 s per window (19 h for the 86k OL training windows), so --n_train keeps a random subset
(seed 0) of the train windows; val is streamed in full unless --no_val. Motion, speech share and window lists are
cut to the same rows. Resumable: rows already written are listed in <out>/done_<split>.npy.

usage: python scripts/stream_build.py <src_dense_dir> <out_dir> [--n_train 12000] [--no_val]
"""
import os, sys, csv, json, argparse
from concurrent.futures import ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from speech_to_motion_aligned import read_window, HubertFeaturizer, T_M, FPS_M

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--n_train", type=int, default=12000); ap.add_argument("--no_val", action="store_true")
ap.add_argument("--reuse", default="", help="streamed dir of a smaller subset: its train rows are kept and copied, not recomputed")
args = ap.parse_args()
meta = json.load(open(f"{args.src}/meta.json"))
L = meta["lookahead"]; DUR, S = (T_M + L) / FPS_M, T_M + L
os.makedirs(args.out, exist_ok=True)
json.dump({**meta, "stream": True, "stream_src": args.src, "n_train": args.n_train}, open(f"{args.out}/meta.json", "w"))
feat = HubertFeaturizer()

for split in ["train"] + ([] if args.no_val else ["val"]):
    ws = list(csv.DictReader(open(f"{args.src}/{split}_windows.csv")))
    src_f = np.load(f"{args.src}/{split}_hubert.npy", mmap_mode="r")
    rows = np.arange(len(ws))
    if split == "train" and args.n_train < len(ws):
        rows = np.sort(np.random.default_rng(0).choice(len(ws), args.n_train, replace=False))
        if args.reuse:                                          # the reused rows plus random others
            key = {(w["audio_filename"], w["start_sec"]): i for i, w in enumerate(ws)}
            old = np.array([key[(w["audio_filename"], w["start_sec"])] for w in csv.DictReader(open(f"{args.reuse}/train_windows.csv"))])
            rest = np.setdiff1d(np.arange(len(ws)), old)
            rows = np.sort(np.concatenate([old, np.random.default_rng(1).choice(rest, args.n_train - len(old), replace=False)]))
    np.save(f"{args.out}/{split}_motion.npy", np.load(f"{args.src}/{split}_motion.npy")[rows])
    if os.path.exists(f"{args.src}/{split}_speech.npy"):
        np.save(f"{args.out}/{split}_speech.npy", np.load(f"{args.src}/{split}_speech.npy")[rows])
    with open(f"{args.out}/{split}_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(ws[0].keys())); w.writeheader(); w.writerows(ws[i] for i in rows)
    fn = f"{args.out}/{split}_hubert.npy"
    out = (np.load(fn, mmap_mode="r+") if os.path.exists(fn)
           else np.lib.format.open_memmap(fn, "w+", np.float16, (len(rows), *src_f.shape[1:])))
    done = np.load(f"{args.out}/done_{split}.npy") if os.path.exists(f"{args.out}/done_{split}.npy") else np.zeros(len(rows), bool)
    if split == "train" and args.reuse and not done.any():
        pos = {r: k for k, r in enumerate(rows)}
        old_f = np.load(f"{args.reuse}/train_hubert.npy", mmap_mode="r")
        for j, w in enumerate(csv.DictReader(open(f"{args.reuse}/train_windows.csv"))):
            k = pos[key[(w["audio_filename"], w["start_sec"])]]
            out[k] = old_f[j]; done[k] = True
        out.flush(); np.save(f"{args.out}/done_{split}.npy", done)
        print(f"reused {done.sum()} rows from {args.reuse}", flush=True)
    todo = np.where(~done)[0]
    load = lambda k: read_window(ws[rows[k]]["audio_filename"], float(ws[rows[k]]["start_sec"]), DUR)
    with ThreadPoolExecutor(8) as ex:
        for c in range(0, len(todo), 64):                       # read the next 64 while streaming these
            nxt = list(ex.map(load, todo[c:c + 64])) if c == 0 else fut
            if c + 64 < len(todo):
                fut_iter = ex.map(load, todo[c + 64:c + 128])
            for k, wav in zip(todo[c:c + 64], nxt):
                out[k] = feat.stream(wav, src_f.shape[1], S)
                done[k] = True
            fut = list(fut_iter) if c + 64 < len(todo) else None
            if c % 512 == 0 or c + 64 >= len(todo):
                out.flush(); np.save(f"{args.out}/done_{split}.npy", done)
                print(f"{split} {done.sum()}/{len(rows)}", flush=True)
print("done", flush=True)
