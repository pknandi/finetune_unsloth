"""D2: look-ahead of Super Star's audio input (its own tokenizer: HuBERT-base over up to 100 s chunks, layer 9,
k-means 500, as scripts/get_speech_code_beat2.py / demo.load_audio_input_tokenize). For a few test windows the audio
after a cut tau is replaced by noise, the tokens are recomputed the same way, and the earliest changed token gives
the look-ahead tau - t_first (50 tokens / s). Any model fed these tokens inherits this look-ahead, whatever its
own causality. Runs in the Super Star venv from the super_star_embody directory.

usage: (cd <super_star_embody> && /data2/home/mubtasim/venvs/superstar/bin/python /home/mubtasim/speech-motion/scripts/superstar_leak.py [n=5])
"""
import sys, glob, os
sys.path.insert(0, ".")
import numpy as np, torch, joblib, soundfile as sf
from fairseq import checkpoint_utils

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["model_files/hubert_models/hubert_base_ls960.pt"])
hub = models[0].eval().cuda()
km = joblib.load("model_files/hubert_models/hubert_base_ls960_L9_km500.bin")


def tokens(wav):
    with torch.no_grad():
        f = hub.extract_features(source=torch.as_tensor(wav, dtype=torch.float32, device="cuda")[None], padding_mask=None,
                                 output_layer=9)[0]
    return km.predict(f[0].cpu().numpy())


wavs = sorted(glob.glob("/data2/home/mubtasim/ext/superstar_embody/wave16k/*.wav"))
rng = np.random.default_rng(0)
worst = -1e9; per = []
for p in [wavs[i] for i in rng.choice(len(wavs), n, replace=False)]:
    w, sr = sf.read(p, dtype="float32"); w = w if w.ndim == 1 else w.mean(1)
    w = w[:sr * 12]
    base = tokens(w)
    for tau in (3.0, 5.0, 7.0):
        if len(w) < (tau + 1) * sr:
            continue
        w2 = w.copy(); c = int(tau * sr)
        w2[c:] = np.random.default_rng(0).normal(0, w.std() + 1e-3, len(w) - c).astype(np.float32)
        t2 = tokens(w2)
        diff = np.where(base[:len(t2)] != t2[:len(base)])[0]
        la = tau - diff[0] / 50.0 if len(diff) else -tau
        per.append(round(float(la), 2)); worst = max(worst, la)
print(f"SSLEAK lookahead_s={worst:.2f} per-cut={per}", flush=True)
