"""Speech-onset anticipation targets (2026-11-07, contribution: onset anticipation). For every window of a streamed data
dir (<dir>/{split}_windows.csv), the person's loudness over the model's audio slots (T_M + L = 158 slots at 15 fps,
10.53 s from start_sec), a per-window speech threshold as scripts/reaction_latency.py (median + 0.5 x (90th pct -
median)), and per slot s a class: 0 = speaking at s; k = 1..15 = silent at s and the next speech starts k slots later
(k / 15 s); 16 = silent and no speech in the next second; -1 = unknown (the next second runs past the window end).
These labels look into the future, so they are used only as a training target (the model predicts them causally).
Writes <dir>/{split}_onset.npy (N, 158) int8.

usage: python scripts/build_onset_labels.py <data dir> [splits=train,val]
"""
import os, sys, csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf

S, H, FPS = 158, 15, 15
ROOTS = ["/data/datasets/embody3d", "/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0", ""]


def path(a):
    if os.path.isabs(a) and os.path.exists(a):
        return a
    for r in ROOTS:
        p = os.path.join(r, a)
        if os.path.exists(p):
            return p
    return a


def labels(w):
    a, st = path(w["audio_filename"]), float(w["start_sec"])
    try:
        sr = sf.info(a).samplerate
        wav, _ = sf.read(a, start=int(st * sr), frames=int(S / FPS * sr) + 1, dtype="float32", always_2d=True)
    except Exception:
        return np.full(S, -1, np.int8)
    wav = wav.mean(1); hop = sr // FPS; k = min(len(wav) // hop, S)
    if k < S // 2:
        return np.full(S, -1, np.int8)
    loud = np.log(np.sqrt((wav[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10))
    thr = np.median(loud) + 0.5 * (np.percentile(loud, 90) - np.median(loud))
    on = loud > thr
    y = np.full(S, -1, np.int8)
    for s in range(k):
        if on[s]:
            y[s] = 0
            continue
        nxt = np.where(on[s + 1:min(s + 1 + H, k)])[0]
        if len(nxt):
            y[s] = nxt[0] + 1
        elif s + H < k:
            y[s] = H + 1
    return y


if __name__ == "__main__":
    d = sys.argv[1]
    for split in (sys.argv[2] if len(sys.argv) > 2 else "train,val").split(","):
        ws = list(csv.DictReader(open(f"{d}/{split}_windows.csv")))
        with ProcessPoolExecutor(24) as ex:
            Y = np.stack(list(ex.map(labels, ws, chunksize=64)))
        np.save(f"{d}/{split}_onset.npy", Y)
        known = Y[Y >= 0]
        print(f"{d} {split}: {len(Y)} windows, unknown {np.mean(Y < 0):.3f}, speaking {np.mean(known == 0):.3f}, "
              f"onset<=1s {np.mean((known > 0) & (known <= H)):.3f}, silent {np.mean(known == H + 1):.3f}, "
              f"all-unknown windows {np.mean((Y < 0).all(1)):.3f}", flush=True)
