"""BEAT2 data in the Embody-3D tokenizer, for fine-tuning an Embody-trained model on BEAT2 (a variant that
uses extra data). build_dense.py already encoded the BEAT2 train windows with --tok_dir (the Embody
tokenizer); its val_motion.npy is a placeholder, so val windows are encoded here (as retokenize_dense.py).

usage: python scripts/beat2_emb_data.py <dense_dir> <out_dir>
"""
import os, sys, csv, json
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion

src, out = sys.argv[1:3]
meta = json.load(open(f"{src}/meta.json"))
tok_dir, Q, T_M = meta["tokenizer_dir"], meta["num_quantizers"], 150
os.makedirs(out, exist_ok=True)
for f in ("train_hubert.npy", "val_hubert.npy", "train_windows.csv", "val_windows.csv", "train_motion.npy"):
    if not os.path.exists(f"{out}/{f}"):
        os.symlink(os.path.abspath(f"{src}/{f}"), f"{out}/{f}")
norm = Normalizer(); norm.load(f"{tok_dir}/normalizer.npz")
tok = VQVAETokenizer(num_quantizers=Q); tok.load(f"{tok_dir}/tokenizer.pt")
va = []
for r in csv.DictReader(open(f"{src}/val_windows.csv")):
    m = load_smplx_sequence(os.path.dirname(os.path.dirname(r["audio_filename"])))
    f0 = round(float(r["start_sec"]) * 30)
    c = tok.encode(norm.transform(preprocess_motion(m[f0:f0 + 2 * T_M + 16], norm.orient_repr, norm.pose_repr)))[:T_M]
    o = np.full((T_M, Q), -1, np.int16); o[:len(c)] = c
    va.append(o)
np.save(f"{out}/val_motion.npy", np.stack(va))
json.dump(meta, open(f"{out}/meta.json", "w"))
print(f"{out}: val {len(va)} windows, tokenizer {tok_dir}", flush=True)
