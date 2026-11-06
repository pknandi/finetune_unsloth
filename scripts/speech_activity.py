"""Share of each 10 s window in which the person's own (separated) audio track is active, for
splitting windows into speaking / listening. Per 50 ms frame: active if its level is at least 20 dB
over the window's 10th-percentile frame level and above -55 dBFS (separated tracks are near digital
silence between turns). An energy measure, not a VAD: laughs and noise count as active.

usage: python scripts/speech_activity.py <data_dir> [<data_dir> ...]
Writes <data_dir>/{train,val}_speech.npy (float32, one value per row of {train,val}_windows.csv).
"""
import os, sys, csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf

DUR, HOP = 10.0, 0.05


def active(row):
    path, start = row
    try:
        sr = sf.info(path).samplerate
        w, _ = sf.read(path, start=int(start * sr), frames=int(DUR * sr), dtype="float32", always_2d=True)
    except Exception:
        return np.nan
    w = w.mean(1); h = int(HOP * sr); n = len(w) // h
    if n == 0:
        return np.nan
    db = 10 * np.log10((w[:n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    return float(((db > np.percentile(db, 10) + 20) & (db > -55)).mean())


if __name__ == "__main__":
    for d in sys.argv[1:]:
        for split in ("val", "train"):
            out = f"{d}/{split}_speech.npy"
            if os.path.exists(out) or not os.path.exists(f"{d}/{split}_windows.csv"):
                continue
            rows = [(r["audio_filename"], float(r["start_sec"])) for r in csv.DictReader(open(f"{d}/{split}_windows.csv"))]
            with ProcessPoolExecutor(24) as ex:
                s = np.array(list(ex.map(active, rows, chunksize=32)), np.float32)
            np.save(out, s)
            print(f"{out}: n={len(s)} nan={np.isnan(s).sum()} median={np.nanmedian(s):.2f} "
                  f"<0.2 {np.nanmean(s < 0.2):.3f} >=0.4 {np.nanmean(s >= 0.4):.3f}", flush=True)
