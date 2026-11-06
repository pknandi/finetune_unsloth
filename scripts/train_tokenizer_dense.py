"""Motion tokenizer on the dense windows (scripts/build_dense.py), with facing relative to each
window's first frame (orient_repr="6d_rel"). Each 10 s window is preprocessed on its own, exactly
as the generator's data will be, and every epoch draws one random 120-frame (4 s) crop per window.
Starts from an existing tokenizer (--init) so it only has to adapt, not relearn.

usage: python scripts/train_tokenizer_dense.py <dense_data_dir> <save_dir> [--init <tokenizer.pt>]
         [--epochs 40 --batch 64 --hidden 2048 --q 8]
Saves <save_dir>/tokenizer.pt + normalizer.npz (orient_repr 6d_rel); resumes from
<save_dir>/train_checkpoint.pt. Prints "Epoch i/n | Loss: x" per epoch.
"""
import os, sys, csv, argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
from vqvae_motion_tokenizer import (VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion,
                                    feature_channel_weights)

W, CROP, ORIENT = 300, 120, "6d_rel"


def load_rec(job):
    """(motion_dir, [start_sec]) -> list of (300, 315) float16 feature windows (None if short)"""
    mdir, starts = job
    try:
        seq = load_smplx_sequence(mdir)
    except Exception:
        return [None] * len(starts)
    out = []
    for s in starts:
        m = seq[round(s * 30):round(s * 30) + W]
        out.append(preprocess_motion(m, ORIENT, "6d").astype(np.float16) if len(m) == W else None)
    return out


class Crops(torch.utils.data.Dataset):
    """one random CROP-frame crop per window per epoch, normalized on the fly"""
    def __init__(self, x, mean, std):
        self.x, self.mean, self.std = x, mean.astype(np.float32), std.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        k = np.random.randint(0, W - CROP + 1)
        return (self.x[i, k:k + CROP].astype(np.float32) - self.mean) / self.std   # VQVAETokenizer.fit wraps it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("save")
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=2048); ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--hand_weight", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.save, exist_ok=True)
    cache = f"{args.save}/train_feats.npy"
    if os.path.exists(cache):
        x = np.load(cache, mmap_mode="r")
    else:
        rows = list(csv.DictReader(open(f"{args.data}/train_windows.csv")))
        by_rec = defaultdict(list)
        for r in rows:
            by_rec[os.path.dirname(os.path.dirname(r["audio_filename"]))].append(float(r["start_sec"]))
        jobs = list(by_rec.items())
        with ProcessPoolExecutor(32) as ex:
            res = list(ex.map(load_rec, jobs, chunksize=8))
        feats = [w for ws in res for w in ws if w is not None]
        print(f"{len(feats)}/{len(rows)} windows loaded from {len(jobs)} recordings", flush=True)
        x = np.stack(feats); del feats
        np.save(cache, x)
    norm = Normalizer(ORIENT, "6d")
    norm.fit(x[np.random.default_rng(0).choice(len(x), min(4000, len(x)), replace=False)].astype(np.float32).reshape(-1, x.shape[2]))
    norm.save(f"{args.save}/normalizer.npz")

    tok = VQVAETokenizer(n_clusters=1024, num_quantizers=args.q, hidden_dim=args.hidden, input_dim=x.shape[2])
    if args.init and not os.path.exists(f"{args.save}/train_checkpoint.pt"):
        tok.load(args.init)
        print(f"init from {args.init}", flush=True)
    tok.fit(Crops(x, norm.mean[0], norm.std[0]), epochs=args.epochs, batch_size=args.batch,
            checkpoint_path=f"{args.save}/train_checkpoint.pt",
            channel_weights=feature_channel_weights("6d", args.hand_weight))
    tok.save(f"{args.save}/tokenizer.pt")
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
