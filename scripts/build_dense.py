"""Dense training data for speech_to_motion_small.py / aligned_eval.py. The run's aligned_data uses ONE
10 s window per recording (the speechiest/most-moving one, stage 2), i.e. 26 h of the ~255 h of
training audio. This takes every 10 s window (stride --stride s) of every training recording whose
audio RMS is at least the recording's median window RMS (the same gate stage 2 applies before
ranking by motion energy), so silent listening stretches stay out.
  motion  tokens exactly as stage 4 (speech_to_motion_pipeline.build_joint_jsonl): slice from the
          window start, preprocess_motion (translation relative to the window start), normalize, encode
  audio   HuBERT-large layer 12 over the window + lookahead (as speech_to_motion_aligned.build_data),
          then adjacent 50 fps frames averaged to 25 fps (half the disk; the small model pools to
          15 fps anyway)
Val = the same windows as <src_data>/val_windows.csv (motion copied, HuBERT recomputed at 25 fps), so
numbers stay comparable with models trained on <src_data>.

usage: python scripts/build_dense.py <run_dir> <src_data_dir> <out_dir> [--stride 10]
"""
import os, sys, csv, json, argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("src_data"); ap.add_argument("out")
ap.add_argument("--stride", type=float, default=10.0)
ap.add_argument("--gate", default="upper", choices=["upper", "lower", "none"],
                help="upper: windows at or above the recording's median RMS (default); lower: the rest; none: all")
ap.add_argument("--tok_dir", default="", help="tokenizer dir (default <run_dir>/motion_tokenizer_artifacts)")
ap.add_argument("--max_recs", type=int, default=0, help="smoke test: first N recordings only (val is cut to 32)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
meta = json.load(open(f"{args.src_data}/meta.json"))
Q, L = meta["num_quantizers"], meta["lookahead"]
T_M, FPS_M, FPS_RAW = 150, 15, 30
DUR = (T_M + L) / FPS_M                       # audio seconds per window (window + lookahead)
NF50 = int(DUR * 50) - 1                      # 50 fps HuBERT frames (529 at L=8)
NF = NF50 // 2                                # 25 fps frames stored


def candidates(rec):
    """(audio, motion_dir) -> [start_sec] of windows passing the RMS gate."""
    audio, mdir = rec
    try:
        info = sf.info(audio)
        wav, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        from vqvae_motion_tokenizer import load_smplx_sequence
        n_mot = len(load_smplx_sequence(mdir))
    except Exception:
        return []
    starts, rms = [], []
    s = 0.0
    while s + DUR <= len(wav) / sr and round(s * FPS_RAW) + 2 * T_M <= n_mot:
        seg = wav[int(s * sr):int((s + 10) * sr)]
        starts.append(s); rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
        s += args.stride
    if not starts:
        return []
    gate = np.median(rms)
    if args.gate == "none":
        return starts
    if args.gate == "lower":
        return [st for st, r in zip(starts, rms) if not (r >= gate and r > 0)]
    return [st for st, r in zip(starts, rms) if r >= gate and r > 0]


def main():
    import torch
    from speech_to_motion_aligned import read_window, HubertFeaturizer
    from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, load_smplx_sequence, preprocess_motion
    mapping = {r["audio_filename"]: r["motion_dirname"]
               for r in csv.DictReader(open(f"{args.run_dir}/tokenizer_dataset_mapping.csv"))}
    src_tr = list(csv.DictReader(open(f"{args.src_data}/train_windows.csv")))
    recs = sorted({(r["audio_filename"], mapping.get(r["audio_filename"],
                    os.path.dirname(os.path.dirname(r["audio_filename"])))) for r in src_tr})
    if args.max_recs:
        recs = recs[:args.max_recs]
    with ProcessPoolExecutor(32) as ex:
        starts = list(ex.map(candidates, recs, chunksize=8))
    wins = [(a, m, s) for (a, m), ss in zip(recs, starts) for s in ss]
    print(f"{len(recs)} recordings -> {len(wins)} windows ({len(wins) * 10 / 3600:.1f} h; source {len(src_tr)})", flush=True)

    tok_dir = args.tok_dir or f"{args.run_dir}/motion_tokenizer_artifacts"
    tok = VQVAETokenizer(num_quantizers=Q); tok.load(f"{tok_dir}/tokenizer.pt")
    norm = Normalizer(); norm.load(f"{tok_dir}/normalizer.npz")
    mot = np.full((len(wins), T_M, Q), -1, np.int16)
    cur, seq = None, None
    for i, (a, mdir, s) in enumerate(wins):
        if mdir != cur:
            cur, seq = mdir, load_smplx_sequence(mdir)
        off = round(s * FPS_RAW)
        m = norm.transform(preprocess_motion(seq[off:off + 2 * T_M + 16], norm.orient_repr, norm.pose_repr))
        c = tok.encode(m)[:T_M]
        mot[i, :len(c)] = c
        if i % 5000 == 0:
            print(f"motion {i}/{len(wins)}", flush=True)
    np.save(f"{args.out}/train_motion.npy", mot)
    with open(f"{args.out}/train_windows.csv", "w", newline="") as f:
        csv.writer(f).writerows([("audio_filename", "start_sec")] + [(a, s) for a, _, s in wins])

    feat = HubertFeaturizer()

    def hubert(split, ws):
        out = np.lib.format.open_memmap(f"{args.out}/{split}_hubert.npy", "w+", np.float16, (len(ws), NF, 1024))
        with ThreadPoolExecutor(16) as ex:
            nxt = ex.map(lambda w: read_window(w[0], w[1], DUR), ws[0:16])
            for i in range(0, len(ws), 16):
                wavs = list(nxt)
                if i + 16 < len(ws):
                    nxt = ex.map(lambda w: read_window(w[0], w[1], DUR), ws[i + 16:i + 32])
                h = feat(np.stack(wavs))[:, :2 * NF].astype(np.float32)
                out[i:i + len(wavs)] = ((h[:, 0::2] + h[:, 1::2]) / 2).astype(np.float16)
                if i % 3200 == 0:
                    print(f"{split} hubert {i}/{len(ws)}", flush=True)
        out.flush()

    va = [(r["audio_filename"], float(r["start_sec"])) for r in csv.DictReader(open(f"{args.src_data}/val_windows.csv"))]
    nva = 32 if args.max_recs else len(va)
    va = va[:nva]
    hubert("val", va)
    np.save(f"{args.out}/val_motion.npy", np.load(f"{args.src_data}/val_motion.npy")[:nva])
    with open(f"{args.out}/val_windows.csv", "w", newline="") as f:
        csv.writer(f).writerows([("audio_filename", "start_sec")] + va)
    hubert("train", [(a, s) for a, _, s in wins])
    json.dump({**meta, "hubert_fps": 25, "dense_stride": args.stride, "src_data": args.src_data, "gate": args.gate,
               "tokenizer_dir": tok_dir, "orient_repr": norm.orient_repr},
              open(f"{args.out}/meta.json", "w"))
    print(f"built {args.out}: {len(wins)} train windows", flush=True)


if __name__ == "__main__":
    main()
