"""Partner audio for conversational motion (contribution P): for every window of a dense / streamed data dir, the
summed separated audio of the OTHER participants of the same Embody-3D session, as causal log-mel features aligned
with the HuBERT frames (25 fps): frame k is the 80-band log-mel of the 40 ms of audio ending at (k + 1) * 40 ms, so it
holds no audio after its own frame (center=False, one hop per frame). Windows with no other participant get zeros.
Writes <dir>/<split>_partner.npy (N, n_frames, 80) float16 next to <split>_hubert.npy.

usage: python scripts/build_partner.py <data_dir> [--workers 32] [--splits train,val]
"""
import os, sys, csv, glob, argparse
from multiprocessing import Pool
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np

SR, HOP, N_MELS = 16000, 640, 80
_MEL = None


def partners(path):
    """separated audio files of the other participants of this window's session"""
    pdir = os.path.dirname(os.path.dirname(path))                  # .../<session>/<pid>/audio_separated/x.wav
    sdir, me = os.path.dirname(pdir), os.path.basename(pdir)
    return [p for p in glob.glob(f"{sdir}/*/audio_separated/*.wav") if os.path.basename(os.path.dirname(os.path.dirname(p))) != me]


def feats(job):
    path, start, nf = job
    global _MEL
    import librosa
    from speech_to_motion_aligned import read_window
    if _MEL is None:
        _MEL = librosa.filters.mel(sr=SR, n_fft=HOP, n_mels=N_MELS)
    others = partners(path)
    if not others:
        return np.zeros((nf, N_MELS), np.float16), 0
    wav = sum(read_window(p, start, nf * HOP / SR) for p in others)
    frames = wav[:nf * HOP].reshape(nf, HOP) * np.hanning(HOP)            # frame k = audio [k, k + 1) * 40 ms
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(spec @ _MEL.T + 1e-6).astype(np.float16), len(others)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    for split in args.splits.split(","):
        ws = list(csv.DictReader(open(f"{args.data}/{split}_windows.csv")))
        nf = np.load(f"{args.data}/{split}_hubert.npy", mmap_mode="r").shape[1]
        out = np.lib.format.open_memmap(f"{args.data}/{split}_partner.npy.tmp", "w+", np.float16, (len(ws), nf, N_MELS))
        n_none = 0
        with Pool(args.workers) as pool:
            for i, (f, k) in enumerate(pool.imap(feats, [(w["audio_filename"], float(w["start_sec"]), nf) for w in ws], chunksize=16)):
                out[i] = f; n_none += k == 0
                if i % 2000 == 0:
                    print(f"{split} {i}/{len(ws)} (no partner so far: {n_none})", flush=True)
        out.flush(); del out
        os.replace(f"{args.data}/{split}_partner.npy.tmp", f"{args.data}/{split}_partner.npy")
        print(f"wrote {args.data}/{split}_partner.npy: {len(ws)} windows, {n_none} without a partner", flush=True)
