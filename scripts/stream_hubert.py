"""Online-honest audio features for the 60 Embody-3D eval windows (scripts/aligned_eval.py: random.seed(0),
random.sample(range(n_val), 60)). The strict online model (--causal_audio) masks audio slots after frame t, but the
HuBERT features it was scored on come from one bidirectional pass over the whole window, so slot t already carries
future audio. HubertFeaturizer.stream recomputes each slot from the audio up to its end only.
Writes <out>/ = <data> with val_hubert.npy replaced (those 60 rows streamed, the rest unchanged), other files
symlinked, so aligned_eval.py can score any checkpoint on it. --check: on one window, change the audio after
slot 80 and report how much slots < 80 move (should be 0) for the old and the streamed features.

usage: python scripts/stream_hubert.py <data_dir> <out_dir> [--check]
"""
import os, sys, csv, json, random, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from speech_to_motion_aligned import read_window, HubertFeaturizer, T_M, FPS_M

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--check", action="store_true")
args = ap.parse_args()
L = json.load(open(f"{args.data}/meta.json"))["lookahead"]
DUR, S = (T_M + L) / FPS_M, T_M + L
va_w = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))
va_f = np.load(f"{args.data}/val_hubert.npy", mmap_mode="r")
NF = va_f.shape[1]
random.seed(0); idx = sorted(random.sample(range(len(va_w)), min(60, len(va_w))))
feat = HubertFeaturizer()


def pooled(wav):
    h = feat(wav[None])[0, :2 * NF].astype(np.float32)
    return (h[0::2] + h[1::2]) / 2


def slot_frames(j):                                   # 25 fps frames before slot j's first frame
    return j * NF // S


if args.check:
    w = read_window(va_w[idx[0]]["audio_filename"], float(va_w[idx[0]]["start_sec"]), DUR)
    w2 = w.copy(); cut = round(80 * 16000 / FPS_M); w2[cut:] = np.random.default_rng(0).normal(0, w.std(), len(w) - cut)
    a = slot_frames(80)
    old = np.abs(pooled(w)[:a] - pooled(w2)[:a]).max()
    new = np.abs(feat.stream(w, NF, S)[:a].astype(np.float32) - feat.stream(w2, NF, S)[:a].astype(np.float32)).max()
    ref = np.abs(pooled(w)[:a]).mean()
    full = np.abs(pooled(w) - feat.stream(w, NF, S).astype(np.float32)).mean()
    print(f"CHECK future audio changed after slot 80: max change of frames < slot 80: old {old:.4f}, streamed {new:.4f} "
          f"(mean |feature| {ref:.3f}); old vs streamed, same audio: mean |diff| {full:.3f}", flush=True)
    sys.exit(0)

os.makedirs(args.out, exist_ok=True)
for f in os.listdir(args.data):
    if f != "val_hubert.npy" and not os.path.exists(f"{args.out}/{f}"):
        os.symlink(os.path.abspath(f"{args.data}/{f}"), f"{args.out}/{f}")
out = np.lib.format.open_memmap(f"{args.out}/val_hubert.npy", "w+", np.float16, va_f.shape)
out[:] = va_f
for k, i in enumerate(idx):
    out[i] = feat.stream(read_window(va_w[i]["audio_filename"], float(va_w[i]["start_sec"]), DUR), NF, S)
    if k % 10 == 0:
        print(f"streamed {k + 1}/{len(idx)}", flush=True)
out.flush()
print(f"wrote {args.out}/val_hubert.npy ({len(idx)} rows streamed)", flush=True)
