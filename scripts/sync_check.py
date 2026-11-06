"""Speech-motion sync check on raw data (no model): per 10 s window, the Pearson correlation over time
between the subject's audio loudness (log RMS, 30 fps) and the speed of their upper body, hands and
head (frame-to-frame change of SMPL-X axis-angle pose), at lags of -1..+1 s. The null is the same
statistic with each window's audio paired with another window's motion. If real pairs don't beat
the null, the audio carries no timing information about the motion for any model to learn.

usage: python scripts/sync_check.py <run_dir> [n_windows=600] [split=train]
Last lines: SYNC part=<p> r0=<corr at lag 0> best_r=<max over lags> best_lag_s=<lag, + = motion after audio> null_r0= null_sd=
"""
import os, sys, csv
from concurrent.futures import ProcessPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import soundfile as sf
from vqvae_motion_tokenizer import load_smplx_sequence

run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
split = sys.argv[3] if len(sys.argv) > 3 else "train"
FPS, WIN, LAGS = 30, 300, np.arange(-30, 31, 3)
# SMPL-X body_pose joints (21, axis-angle) start at column 3; hands follow body_pose at column 66
PARTS = {"upper": [12, 13, 15, 16, 17, 18, 19, 20], "head": [11, 14], "hands": None}

motion_dir = {r["audio_filename"]: r["motion_dirname"]
              for r in csv.DictReader(open(f"{run}/tokenizer_dataset_mapping.csv"))}
rows = list(csv.DictReader(open(f"{run}/speech_motion_{split}_audio_windows.csv")))
rng = np.random.default_rng(0)
rows = [rows[i] for i in sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))]


def one(r):
    try:
        sr = sf.info(r["audio_filename"]).samplerate
        start = float(r["start_sec"])
        wav, _ = sf.read(r["audio_filename"], start=int(start * sr), frames=10 * sr, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        hop = sr // FPS
        k = min(len(wav) // hop, WIN)
        rms = np.sqrt((wav[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10)
        loud = np.log(rms)
        m = load_smplx_sequence(motion_dir[r["audio_filename"]])
        f0 = round(start * FPS)
        m = m[f0:f0 + k + 1]
        if len(m) < k + 1:
            return None
        body = m[:, 3:66].reshape(len(m), 21, 3)
        sp = {}
        for p, joints in PARTS.items():
            x = m[:, 66:156] if joints is None else body[:, joints].reshape(len(m), -1)
            sp[p] = np.linalg.norm(np.diff(x, axis=0), axis=1)[:k]
        return loud, sp
    except Exception:
        return None


def corr(a, b, lag):
    """corr(a[t], b[t + lag]): positive lag = motion after audio."""
    if lag > 0: a, b = a[:-lag], b[lag:]
    elif lag < 0: a, b = a[-lag:], b[:lag]
    if a.std() < 1e-8 or b.std() < 1e-8: return np.nan
    return np.corrcoef(a, b)[0, 1]


with ProcessPoolExecutor(16) as ex:
    data = [d for d in ex.map(one, rows, chunksize=4) if d is not None]
print(f"{len(data)} windows")
for p in PARTS:
    real = np.array([[corr(l, s[p], lag) for lag in LAGS] for l, s in data])
    curve = np.nanmean(real, 0)
    null = []
    for shift in range(1, 21):
        null.append(np.nanmean([corr(data[i][0], data[(i + shift * 7) % len(data)][1][p][:len(data[i][0])][:len(data[i][0])], 0)
                                if len(data[(i + shift * 7) % len(data)][1][p]) >= len(data[i][0]) else np.nan
                                for i in range(len(data))]))
    b = int(np.nanargmax(curve))
    frac_pos = np.nanmean(real[:, LAGS.tolist().index(0)] > 0)
    print("lags(s) " + " ".join(f"{l / FPS:+.1f}:{c:+.3f}" for l, c in zip(LAGS, curve)))
    print(f"SYNC part={p} r0={curve[LAGS.tolist().index(0)]:+.3f} best_r={curve[b]:+.3f} best_lag_s={LAGS[b] / FPS:+.2f} "
          f"null_r0={np.nanmean(null):+.3f} null_sd={np.nanstd(null):.3f} frac_windows_r0_pos={frac_pos:.2f}", flush=True)
