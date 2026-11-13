"""Embody-3D in the BEAT2 layout Super Star reads (lom/data/mixed_dataset/dataset_a2m*.py), so Super Star can be
trained and tested on our data like on BEAT2. Runs in the EMAGE venv.

  train  every training recording our model saw (the dense windows' recordings, upper + lower RMS halves), whole
  val    60 of the 531 held-out val windows (random.seed(1), not among the 60 test ones): checkpoint selection
  test   the 60 held-out windows our models and EMAGE are scored on (scripts/aligned_eval.py: random.seed(0),
         random.sample(range(531), 60) on <val_data>/val_windows.csv), 10 s each

Each recording / window is turned to face +Z at its first frame with x, z starting at 0 (as scripts/emage_export.py
for the EMAGE VQ we reuse); scripts/superstar_embody_bridge.py turns test outputs back. Jaw / eyes / expressions are
zero (not in Embody-3D), betas padded to 300, audio resampled to 16 kHz. Ids are <subject number>_<subject>_<k>
(Super Star reads the speaker as an int before the first "_"); TextGrids are one empty interval (only metadata).
Writes <out>/{smplxflame_30,wave16k,textgrid}/, train_test_split.csv, subjects.csv, windows.csv (id -> source window).

--eval_sec 12.5 --only_eval: rewrite only the val / test windows, 12.5 s long (Super Star's test loop makes 4 seed
frames + 116-frame chunks, so 10 s gave only 236 frames; it is causal, so frames 0-300 still see only the window's
first 10 s of audio). Windows near a recording's end are padded (silent audio, last pose held).

usage: python scripts/superstar_embody_data.py <out_root> [--max_train N] [--eval_sec 12.5 --only_eval]
"""
import os, sys, csv, glob, random, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--train_lists", default="outputs/run-full-q8-oct21-dense/train_windows.csv,outputs/run-full-dense-rel-lower-oct30/train_windows.csv")
ap.add_argument("--val_data", default="outputs/run-full-dense-rel-oct28")
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--eval_sec", type=float, default=10.0)
ap.add_argument("--only_eval", action="store_true")
ap.add_argument("--groups", default="", help="2026-11-08: add the fresh eval groups (e.g. 1,2,3,4,5,6,7) to an existing root")
args = ap.parse_args()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PARTS = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]
FPS, SR, W = 30, 16000, 300
TG = ('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {d}\ntiers? <exists>\nsize = 1\n'
      'item []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = {d}\n'
      '        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = {d}\n            text = ""\n')


def load(sd):
    a = {p: np.load(glob.glob(f"{sd}/smplx_mesh_{p}/*.npy")[0]).astype(np.float32) for p in PARTS + ["betas"]}
    n = min(len(a[p]) for p in PARTS)
    m = np.concatenate([a[p][:n].reshape(n, -1) for p in PARTS], 1)          # (n, 159)
    return m, a["betas"].reshape(-1)[:10]


def face_z(m):
    """turn so frame 0 faces +Z, x / z start at 0"""
    f = R.from_rotvec(m[0, :3]).apply([0, 0, 1.0]); yaw = np.arctan2(f[0], f[2])
    rot = R.from_euler("y", -yaw)
    o = m.copy()
    o[:, :3] = (rot * R.from_rotvec(m[:, :3])).as_rotvec()
    t = m[:, 156:159] - m[0, 156:159] * np.array([1, 0, 1], np.float32)
    o[:, 156:159] = rot.apply(t)
    return o


def write(job):
    rid, audio, start, sec = job
    try:
        sd = os.path.dirname(os.path.dirname(audio))
        m, betas = load(sd)
        f0 = round(start * FPS)
        n = len(m) - f0 if sec is None else round(sec * FPS)
        wav, _ = librosa.load(audio, sr=SR, offset=start, duration=None if sec is None else sec)
        want = n
        n = min(n, len(m) - f0, len(wav) * FPS // SR)
        if n < 64:
            return rid, 0
        m = face_z(m[f0:f0 + n])
        if sec is not None and n < want:                     # eval window near the end: pad to the full length
            m = np.concatenate([m, np.repeat(m[-1:], want - n, 0)])
            wav = np.pad(wav[:n * SR // FPS], (0, (want - n) * SR // FPS))
            n = want
        poses = np.zeros((n, 165), np.float32)
        poses[:, :66] = m[:, :66]; poses[:, 75:165] = m[:, 66:156]
        b = np.zeros(300, np.float32); b[:10] = betas
        np.savez(f"{args.out}/smplxflame_30/{rid}.npz", poses=poses, trans=m[:, 156:159], betas=b,
                 expressions=np.zeros((n, 100), np.float32), model="smplx2020", gender="neutral", mocap_frame_rate=FPS)
        sf.write(f"{args.out}/wave16k/{rid}.wav", wav[:n * SR // FPS], SR, subtype="PCM_16")
        open(f"{args.out}/textgrid/{rid}.TextGrid", "w").write(TG.format(d=n / FPS))
        return rid, n
    except Exception as e:
        print(f"{rid}: {e}", flush=True)
        return rid, 0


if __name__ == "__main__":
    if args.groups:
        # fresh eval groups (scripts/aligned_eval.py EVAL_GROUP=k: 60 val windows each, disjoint from the default group),
        # written as <sid>_<subject>_g<val index> at --eval_sec, appended to train_test_split.csv with type "groups"
        # (ignored by Super Star's train / val / test splits) and listed per group in <out>/groups/g<k>.txt; Super Star
        # reads only these ids with SS_ONLY_IDS (patched lom/data/mixed_dataset/dataset_a2m.py, dataset_mixed_vq.py)
        va = list(csv.DictReader(open(f"{args.val_data}/val_windows.csv")))
        random.seed(0); g0 = sorted(random.sample(range(len(va)), 60))
        rest = [i for i in np.random.RandomState(0).permutation(len(va)) if i not in set(g0)]
        subj = lambda a: os.path.basename(os.path.dirname(os.path.dirname(a)))
        sid = {r["subject"]: int(r["number"]) for r in csv.DictReader(open(f"{args.out}/subjects.csv"))}
        known = {r["id"] for r in csv.DictReader(open(f"{args.out}/train_test_split.csv"))}
        os.makedirs(f"{args.out}/groups", exist_ok=True)
        jobs, meta = [], []
        for k in map(int, args.groups.split(",")):
            ids = []
            for i in sorted(rest[(k - 1) * 60:k * 60]):
                a, st = va[i]["audio_filename"], float(va[i]["start_sec"])
                if subj(a) not in sid:
                    print(f"skip val {i}: subject {subj(a)} not in subjects.csv", flush=True); continue
                rid = f"{sid[subj(a)]}_{subj(a)}_g{i}"; ids.append(rid)
                if rid not in known:
                    jobs.append((rid, a, st, args.eval_sec)); meta.append((rid, "groups", a, st, i))
            open(f"{args.out}/groups/g{k}.txt", "w").write("\n".join(ids) + "\n")
        with ProcessPoolExecutor(32) as ex:
            done = dict(ex.map(write, jobs, chunksize=4))
        keep = [x for x in meta if done.get(x[0], 0) > 0]
        with open(f"{args.out}/train_test_split.csv", "a", newline="") as f:
            csv.writer(f).writerows([(x[0], x[1]) for x in keep])
        with open(f"{args.out}/windows.csv", "a", newline="") as f:
            csv.writer(f).writerows(keep)
        print(f"groups {args.groups}: wrote {len(keep)} of {len(jobs)} new windows", flush=True)
        sys.exit(0)
    for d in ("smplxflame_30", "wave16k", "textgrid"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    tr = sorted({r["audio_filename"] for f in args.train_lists.split(",") for r in csv.DictReader(open(f))})
    if args.max_train:
        tr = tr[:args.max_train]
    va = list(csv.DictReader(open(f"{args.val_data}/val_windows.csv")))
    random.seed(0); test_idx = sorted(random.sample(range(len(va)), min(60, len(va))))
    rest = [i for i in range(len(va)) if i not in set(test_idx)]
    random.seed(1); val_idx = sorted(random.sample(rest, 60))
    subj = lambda a: os.path.basename(os.path.dirname(os.path.dirname(a)))
    names = sorted({subj(a) for a in tr} | {subj(va[i]["audio_filename"]) for i in test_idx + val_idx})
    sid = {s: k + 1 for k, s in enumerate(names)}
    jobs, meta = [], []
    for k, a in enumerate(tr):
        rid = f"{sid[subj(a)]}_{subj(a)}_t{k}"; jobs.append((rid, a, 0.0, None)); meta.append((rid, "train", a, 0.0, -1))
    for split, idx in (("val", val_idx), ("test", test_idx)):
        for i in idx:
            a, s = va[i]["audio_filename"], float(va[i]["start_sec"])
            rid = f"{sid[subj(a)]}_{subj(a)}_v{i}"; jobs.append((rid, a, s, args.eval_sec)); meta.append((rid, split, a, s, i))
    if args.only_eval:                                    # keep train files and the split / windows lists as they are
        jobs = [j for j in jobs if j[0].rsplit("_", 1)[1].startswith("v")]
        with ProcessPoolExecutor(32) as ex:
            done = dict(ex.map(write, jobs, chunksize=4))
        print(f"rewrote {sum(v > 0 for v in done.values())} eval windows at {args.eval_sec} s", flush=True)
        sys.exit(0)
    with ProcessPoolExecutor(32) as ex:
        done = dict(ex.map(write, jobs, chunksize=16))
    keep = [x for x in meta if done.get(x[0], 0) > 0]
    with open(f"{args.out}/train_test_split.csv", "w", newline="") as f:
        csv.writer(f).writerows([("id", "type")] + [(x[0], x[1]) for x in keep])
    with open(f"{args.out}/windows.csv", "w", newline="") as f:
        csv.writer(f).writerows([("id", "split", "audio_filename", "start_sec", "val_index")] + keep)
    with open(f"{args.out}/subjects.csv", "w", newline="") as f:
        csv.writer(f).writerows([("number", "subject")] + [(v, k) for k, v in sid.items()])
    for split in ("train", "val", "test"):
        xs = [x for x in keep if x[1] == split]
        print(f"{split}: {len(xs)} recordings, {sum(done[x[0]] for x in xs) / FPS / 3600:.1f} h", flush=True)
    print(f"{len(sid)} subjects; failed {len(meta) - len(keep)}", flush=True)
