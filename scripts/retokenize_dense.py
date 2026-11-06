"""Re-tokenize the dense data with another motion tokenizer (e.g. the 6d_rel one from
scripts/train_tokenizer_dense.py). HuBERT features and window lists are linked, not copied; only
train_motion.npy / val_motion.npy are new. Train windows come from the tokenizer's own feature cache
(<tok_dir>/train_feats.npy, same windows and order as <dense_dir>/train_windows.csv).

usage: python scripts/retokenize_dense.py <dense_dir> <tok_dir> <out_dir>
"""
import os, sys, csv, json
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion

src, tok_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
meta = json.load(open(f"{src}/meta.json"))
Q, T_M = meta["num_quantizers"], 150
for f in ("train_hubert.npy", "val_hubert.npy", "train_windows.csv", "val_windows.csv"):
    if not os.path.exists(f"{out}/{f}"):
        os.symlink(os.path.abspath(f"{src}/{f}"), f"{out}/{f}")
norm = Normalizer(); norm.load(f"{tok_dir}/normalizer.npz")
tok = VQVAETokenizer(num_quantizers=Q); tok.load(f"{tok_dir}/tokenizer.pt")
Q = tok.num_quantizers

# the feature cache groups windows by recording in order of first appearance; check it matches the csv
rows = list(csv.DictReader(open(f"{src}/train_windows.csv")))
by_rec = defaultdict(list)
for k, r in enumerate(rows):
    by_rec[os.path.dirname(os.path.dirname(r["audio_filename"]))].append(k)
order = [k for ks in by_rec.values() for k in ks]
assert order == list(range(len(rows))), "train_feats.npy order differs from train_windows.csv"
x = np.load(f"{tok_dir}/train_feats.npy", mmap_mode="r")
assert len(x) == len(rows), (len(x), len(rows))


def encode(feat):
    c = tok.encode(norm.transform(np.asarray(feat, np.float32)))[:T_M]
    o = np.full((T_M, Q), -1, np.int16); o[:len(c)] = c
    return o


tr = np.stack([encode(x[i]) for i in range(len(x))])
np.save(f"{out}/train_motion.npy", tr)
va_rows = list(csv.DictReader(open(f"{src}/val_windows.csv")))
va = []
for r in va_rows:
    m = load_smplx_sequence(os.path.dirname(os.path.dirname(r["audio_filename"])))
    f0 = round(float(r["start_sec"]) * 30)
    va.append(encode(preprocess_motion(m[f0:f0 + 2 * T_M], norm.orient_repr, norm.pose_repr)))
np.save(f"{out}/val_motion.npy", np.stack(va))
json.dump({**meta, "num_quantizers": Q, "tokenizer_dir": tok_dir, "orient_repr": norm.orient_repr},
          open(f"{out}/meta.json", "w"))
print(f"{out}: train {tr.shape}, val {len(va)}; level-0 change rate "
      f"{(tr[:, 1:, 0] != tr[:, :-1, 0]).mean():.3f}", flush=True)
