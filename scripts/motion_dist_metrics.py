"""Distribution metrics for generated motion, which (unlike MPJPE) a still pose cannot win: one speech
clip allows many correct gestures, so a good model should produce motion that LOOKS like real motion
and moves WITH the audio, not motion that matches the one GT take.

Per-frame features (30 fps): body + hand pose (6D, global orientation excluded) projected onto 16
principal components fit on training GT, plus frame-to-frame speed of head, upper body, each hand.
Each dim is standardized by training GT.
  fd           Frechet distance between all generated frames and the same clips' GT frames (0 = same
               distribution; a still pose is far off because its speed is 0 and its pose is one point)
  div_within   mean distance of a clip's frames to that clip's mean (how much it moves), / GT's
  div_between  mean pairwise distance between clip means (do different clips differ), / GT's
  sync         per-clip corr(audio loudness, upper-body speed), both high-passed (1 s moving average
               removed), mean over lags -0.2..+0.2 s
               (GT's own value is the target; ~0 = motion ignores the audio's timing)

usage: python scripts/motion_dist_metrics.py <eval_dir> <data_dir>
       (eval_dir: scripts/aligned_eval.py output, files val<i>_<subject>_<mode>.npy; data_dir: aligned_data)
Last lines: DIST mode=<each mode> fd= div_within= div_between= sync= n=, then mode=other_gt (real motion
of other val clips against these clips' audio) and mode=gt (the target)
"""
import os, sys, re, csv, glob
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import soundfile as sf
from scipy import linalg
from vqvae_motion_tokenizer import load_smplx_sequence, rotvec_to_6d

ev, data = (sys.argv[1:3] if __name__ == "__main__" else (None, os.environ.get("MDM_DATA", "outputs/run-full-q8-oct21/aligned_data")))
FPS, WIN, NPC = 30, 300, 16
LAGS = range(-6, 7)
PARTS = {"head": [11, 14], "upper": [12, 13, 15, 16, 17, 18, 19, 20]}    # SMPL-X body_pose joint ids


def pose6d(m):
    """(T, >=156) axis-angle SMPL-X -> (T, 51*6) body + hands 6D."""
    return rotvec_to_6d(m[:, 3:156].reshape(-1, 3)).reshape(len(m), -1)


def speeds(m):
    body = m[:, 3:66].reshape(len(m), 21, 3)
    out = [np.linalg.norm(np.diff(body[:, j].reshape(len(m), -1), axis=0), axis=1) for j in PARTS.values()]
    out += [np.linalg.norm(np.diff(m[:, a:a + 45], axis=0), axis=1) for a in (66, 111)]
    return np.stack(out, 1)                                                # (T-1, 4)


def gt_window(audio, start):
    m = load_smplx_sequence(os.path.dirname(os.path.dirname(audio)))
    f0 = round(start * FPS)
    return m[f0:f0 + WIN]


def loudness(audio, start, n):
    sr = sf.info(audio).samplerate
    wav, _ = sf.read(audio, start=int(start * sr), frames=10 * sr, dtype="float32", always_2d=True)
    wav = wav.mean(1); hop = sr // FPS
    k = min(len(wav) // hop, n)
    return np.log(np.sqrt((wav[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10))


def highpass(x, w=FPS):
    """subtract a 1 s moving average: slow drifts in loudness and speed otherwise correlate by chance"""
    return x - np.convolve(np.pad(x, w // 2, mode="edge"), np.ones(w) / w, "valid")[:len(x)]


def corr_lags(a, b):
    a, b = highpass(a), highpass(b)
    rs = []
    for lag in LAGS:
        x, y = (a[:-lag], b[lag:]) if lag > 0 else (a[-lag:], b[:lag]) if lag < 0 else (a, b)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        if x.std() > 1e-8 and y.std() > 1e-8:
            rs.append(np.corrcoef(x, y)[0, 1])
    return np.mean(rs) if rs else np.nan


# PCA + standardization from training GT
tr = list(csv.DictReader(open(f"{data}/train_windows.csv")))
rng = np.random.default_rng(0)
ref = [gt_window(tr[i]["audio_filename"], float(tr[i]["start_sec"])) for i in rng.choice(len(tr), 150, replace=False)]
ref = [m for m in ref if len(m) == WIN]
P = np.concatenate([pose6d(m) for m in ref])
mu = P.mean(0)
_, _, Vt = np.linalg.svd(P[rng.choice(len(P), min(20000, len(P)), replace=False)] - mu, full_matrices=False)
V = Vt[:NPC].T
S = np.concatenate([speeds(m) for m in ref])
pc_ref = (P - mu) @ V
f_mu = np.concatenate([pc_ref.mean(0), S.mean(0)]); f_sd = np.concatenate([pc_ref.std(0), S.std(0)]) + 1e-8


def feats(m):
    pc = (pose6d(m) - mu) @ V
    return (np.concatenate([pc[1:], speeds(m)], 1) - f_mu) / f_sd          # (T-1, NPC + 4)


def fd(a, b):
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(c1 @ c2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs.real))


def diversity(clips):
    means = np.stack([c[:, :NPC].mean(0) for c in clips])
    within = np.mean([np.linalg.norm(c[:, :NPC] - c[:, :NPC].mean(0), axis=1).mean() for c in clips])
    between = np.mean([np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))])
    return within, between


if __name__ == "__main__":
    va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
    by_mode = defaultdict(dict)
    for p in glob.glob(f"{ev}/val*_*.npy"):
        mt = re.match(r"val(\d+)_(.+)_([a-z0-9_.]+?)\.npy$", os.path.basename(p))
        if mt and not p.endswith("_ids.npy"):
            by_mode[mt.group(3)][int(mt.group(1))] = p
    idx = sorted(set.intersection(*[set(d) for d in by_mode.values()]))
    gt, loud = {}, {}
    for i in idx:
        a, s = va[i]["audio_filename"], float(va[i]["start_sec"])
        gt[i] = gt_window(a, s); loud[i] = loudness(a, s, WIN)
    gt_f = [feats(gt[i]) for i in idx]
    # speaking / listening split by the share of the window with the person's own audio active
    # (scripts/speech_activity.py -> <data>/val_speech.npy), when available
    subsets = []
    if os.path.exists(f"{data}/val_speech.npy"):
        sp = np.load(f"{data}/val_speech.npy")
        subsets = [("speak", lambda i: sp[i] >= 0.4), ("listen", lambda i: not sp[i] >= 0.4)]
    gw, gb = diversity(gt_f)
    gt_sync = np.nanmean([corr_lags(loud[i][:len(gt[i]) - 1], speeds(gt[i])[:, 1]) for i in idx])
    for mode in sorted(by_mode):
        clips = []
        for i in idx:
            m = np.load(by_mode[mode][i])[:len(gt[i])]
            clips.append(m)
        pf = [feats(m) for m in clips]
        w, b = diversity(pf)
        sync = np.nanmean([corr_lags(loud[i][:len(m) - 1], speeds(m)[:, 1]) for i, m in zip(idx, clips)])
        print(f"DIST mode={mode} fd={fd(np.concatenate(pf), np.concatenate(gt_f)):.2f} div_within={w / gw:.2f} "
              f"div_between={b / gb:.2f} sync={sync:+.3f} n={len(idx)}", flush=True)
        for sub, sel in subsets:
            k = [j for j, i in enumerate(idx) if sel(i)]
            sw = np.mean([np.linalg.norm(pf[j][:, :NPC] - pf[j][:, :NPC].mean(0), axis=1).mean() for j in k])
            gsw = np.mean([np.linalg.norm(gt_f[j][:, :NPC] - gt_f[j][:, :NPC].mean(0), axis=1).mean() for j in k])
            print(f"DISTSUB mode={mode} sub={sub} fd={fd(np.concatenate([pf[j] for j in k]), np.concatenate([gt_f[j] for j in k])):.2f} "
                  f"div_within={sw / gsw:.2f} n={len(k)}", flush=True)
    # real motion from OTHER val clips scored the same way: the fd/sync a realistic model that ignores
    # these clips' audio would get (the floor to compare models against)
    others = [k for k in np.random.default_rng(1).permutation(len(va)) if k not in set(idx)][:len(idx)]
    of, osync = [], []
    for k, i in zip(others, idx):
        m = gt_window(va[k]["audio_filename"], float(va[k]["start_sec"]))
        if len(m) < WIN:
            continue
        of.append(feats(m)); osync.append(corr_lags(loud[i][:WIN - 1], speeds(m)[:, 1]))
    w, b = diversity(of)
    print(f"DIST mode=other_gt fd={fd(np.concatenate(of), np.concatenate(gt_f)):.2f} div_within={w / gw:.2f} "
          f"div_between={b / gb:.2f} sync={np.nanmean(osync):+.3f} n={len(of)}", flush=True)
    print(f"DIST mode=gt fd=0 div_within=1.00 div_between=1.00 sync={gt_sync:+.3f} n={len(idx)}", flush=True)
