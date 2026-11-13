"""Runtime of the online pipeline (D6), batch 1, bf16 autocast where the model uses it, on one GPU: per motion frame
(15 tokens / s, 66.7 ms of real time each) the cost of (1) the streamed HuBERT features for that slot
(HubertFeaturizer.stream: one pass over the audio heard so far), (2) generating the frame's 8 RVQ codes (fixed
guidance, adaptive guidance, energy-guided best-of-K), (3) causal decoding of its two 30 fps poses. Real-time factor
= compute per frame / 66.7 ms (< 1 = faster than real time).

usage: python scripts/measure_latency.py [ckpt=outputs/dt-Db-nov06/step3000] [tok=outputs/tok-dense-rel-causal-nov06]
"""
import os, sys, time, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from speech_to_motion_small import load_trained, generate, T_M
from speech_to_motion_aligned import read_window, HubertFeaturizer, FPS_M
from vqvae_motion_tokenizer import VQVAETokenizer

ck = sys.argv[1] if len(sys.argv) > 1 else "outputs/dt-Db-nov06/step3000"
tok = sys.argv[2] if len(sys.argv) > 2 else "outputs/tok-dense-rel-causal-nov06"
model = load_trained(ck)[0]
w0 = list(csv.DictReader(open("outputs/stream-hubert-nov04/val_windows.csv")))[0]
L = model.L; S = T_M + L; DUR = S / FPS_M
wav = read_window(w0["audio_filename"], float(w0["start_sec"]), DUR)
NF = np.load("outputs/stream-hubert-nov04/val_hubert.npy", mmap_mode="r").shape[1]
feat = HubertFeaturizer()
sync = torch.cuda.synchronize


def timed(fn, reps=3):
    fn(); sync(); t0 = time.time()
    for _ in range(reps):
        fn()
    sync(); return (time.time() - t0) / reps


# (1) streamed features: one HuBERT pass per slot over the audio so far; the cost grows with the context, so take the
# pass at the end of a 10.5 s window (the largest a slot needs within a window)
x = torch.tensor(wav[None], device=feat.device).half()
t_feat = timed(lambda: feat.model(x, output_hidden_states=True))
f = feat.stream(wav, NF, S)
# (2) generation per frame
per = {}
for name, kw in (("fixed g3.5", dict(guidance=3.5)), ("adaptive 1.5-5", dict(adapt=(1.5, 5.0)))):
    t = timed(lambda: generate(model, None, None, f, do_sample=True, temperature=1.3, top_p=1.0, **kw), reps=1)
    per[name] = t / T_M
from energy_decode import generate_energy
causal = np.repeat(np.asarray(f, np.float32), 2, 0)
t = timed(lambda: generate_energy(model, f, causal, causal, tok, K=4, energy="d", head="outputs/energy-head-disc-nov06/energy_head.pt"), reps=1)
per["energy-guided K=4"] = t / T_M
t = timed(lambda: generate_energy(model, f, causal, causal, tok, K=4, energy="d", head="outputs/energy-head-disc-nov06/energy_head.pt",
                                  adapt=(1.5, 5.0)), reps=1)
per["UpBeat (energy K=4 + adaptive)"] = t / T_M                                # the final decoding (mode e4da1.5-5)
for K in (2, 8):                                                                # energy candidate count (2026-11-10)
    t = timed(lambda: generate_energy(model, f, causal, causal, tok, K=K, energy="d", head="outputs/energy-head-disc-nov06/energy_head.pt",
                                      adapt=(1.5, 5.0)), reps=1)
    per[f"energy K={K} + adaptive"] = t / T_M
# (3) causal decoding of one frame's tokens (2 poses), with the window's history
vq = VQVAETokenizer(); vq.load(f"{tok}/tokenizer.pt"); vq.model.eval()
ids = torch.zeros(1, T_M, model.Q, dtype=torch.long, device="cuda")
t_dec = timed(lambda: vq.model.decode_from_tokens(ids)) / T_M
FRAME = 1000 / FPS_M
print(f"LAT streamed HuBERT pass (10.5 s context) {t_feat * 1000:.1f} ms; causal decode {t_dec * 1000:.2f} ms/frame", flush=True)
for name, t in per.items():
    tot = t_feat + t + t_dec
    print(f"LAT {name}: generation {t * 1000:.1f} ms/frame, total {tot * 1000:.1f} ms/frame, real-time factor {tot * 1000 / FRAME:.2f}", flush=True)
print(f"LAT GPU {torch.cuda.get_device_name()}", flush=True)
