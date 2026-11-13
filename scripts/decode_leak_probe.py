"""Audio look-ahead of the whole DECODING path (2026-11-11), not only the generator (scripts/leak_probe.py checks the
generator's teacher-forced logits, so it cannot see look-ahead added by the decoding, e.g. best-of-K chunk selection in
scripts/energy_decode.py). For a few eval windows and cut times tau, the audio after tau is replaced by noise and the
streamed features rebuilt (HubertFeaturizer.stream, as the evals); each decoding mode is run on the clean and the cut
features with the same random seed. The first output slot t whose tokens differ heard audio after tau; slot t is due at
(t + 1) / 15 s, so its look-ahead is tau - (t + 1) / 15 (max over windows and cuts; <= 0 = zero audio look-ahead).

usage: python scripts/decode_leak_probe.py <ckpt_dir> <mode>[,<mode>...] [--head <energy_head.pt>] [--n 3]
  modes as scripts/aligned_eval.py (sample1.3p1f15g3.5, sample1.3p1f15e4da1.5-5, sample1.3p1f15e4dk1a1.5-5, ...)
Prints per mode: DLEAK mode=<> lookahead_s=<max> per-cut=<...>
"""
import os, sys, csv, json, random, re, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from speech_to_motion_aligned import read_window, HubertFeaturizer, T_M, FPS_M
from speech_to_motion_small import load_trained, generate
from energy_decode import generate_energy

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("modes")
ap.add_argument("--head", default=os.environ.get("ENERGY_HEAD", "outputs/energy-head-nov06/energy_head.pt"))
ap.add_argument("--data", default="outputs/run-full-dense-rel-oct28"); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--taus", default="3,5,7")
args = ap.parse_args()
L = json.load(open(f"{args.data}/meta.json"))["lookahead"]
DUR, S = (T_M + L) / FPS_M, T_M + L
va_w = list(csv.DictReader(open(f"{args.data}/val_windows.csv")))
va_m = np.load(f"{args.data}/val_motion.npy")
NF = np.load(f"{args.data}/val_hubert.npy", mmap_mode="r").shape[1]
random.seed(0); idx = sorted(random.sample(range(len(va_w)), 60))[:args.n]
lm, io, bos = load_trained(args.ckpt)
feat = HubertFeaturizer()
RX = r"sample([0-9.]*)(?:p([0-9.]+))?(?:c(\d+))?(?:f(\d+))?(?:g([0-9.]+))?(?:e(\d+)(c|t|T|dr|d|r|o\d+)(?:k(\d+))?)?(?:a([0-9.]+)-([0-9.]+))?(?:s(\d+))?$"


def run(mode, f, m):
    mt = re.match(RX, mode); assert mt and mt.group(7) in (None, "d"), mode
    kw = {"temperature": float(mt.group(1) or 1)}
    if mt.group(4):
        kw["prefix"] = m[:int(mt.group(4))].astype(np.int64)
    if mt.group(5):
        kw["guidance"] = float(mt.group(5))
    if mt.group(9):
        kw["adapt"] = (float(mt.group(9)), float(mt.group(10)))
    torch.manual_seed(int(mt.group(11) or 0)); np.random.seed(0)
    if mt.group(6):
        if mt.group(8):
            kw["C"] = int(mt.group(8)); kw["win"] = 15
        if os.environ.get("ENERGY_PLAN"):                          # 2026-11-12: zero look-ahead planning
            kw["plan"] = tuple(int(x) for x in os.environ["ENERGY_PLAN"].split("x"))
        z = np.zeros((2 * len(f), 1024), np.float32)              # matcher audio: unused by energy "d"
        return generate_energy(lm, f, z, z, None, K=int(mt.group(6)), energy="d", head=args.head, **kw)
    if mt.group(2):
        kw["top_p"] = float(mt.group(2))
    return generate(lm, io, bos, f, do_sample=True, **kw)


modes = args.modes.split(","); taus = [float(x) for x in args.taus.split(",")]
res = {mo: [] for mo in modes}
for i in idx:
    w = read_window(va_w[i]["audio_filename"], float(va_w[i]["start_sec"]), DUR)
    m = va_m[i]; f0 = feat.stream(w, NF, S)
    cut = {}
    for tau in taus:
        w2 = w.copy(); c = int(tau * 16000)
        w2[c:] = np.random.default_rng(0).normal(0, w.std() + 1e-3, len(w) - c)
        cut[tau] = feat.stream(w2, NF, S)
    for mo in modes:
        a = np.asarray(run(mo, f0, m)); a2 = np.asarray(run(mo, f0, m))
        same = bool((a == a2).all())                                     # determinism check (clean vs clean)
        for tau in taus:
            b = np.asarray(run(mo, cut[tau], m))
            d = np.where((a != b).any(1))[0]
            la = tau - (d[0] + 1) / FPS_M if len(d) else -tau
            res[mo].append(round(float(la), 3))
            print(f"DLEAK_ONE val{i} mode={mo} tau={tau} deterministic={same} first_diff_slot={d[0] if len(d) else None} lookahead_s={la:.3f}", flush=True)
for mo in modes:
    print(f"DLEAK ckpt={args.ckpt} mode={mo} lookahead_s={max(res[mo]):.3f} per-cut={res[mo]}", flush=True)
