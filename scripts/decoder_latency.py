"""How far back does a motion token reach in the decoded pose? Change one token at slot s (all RVQ levels) and find the
earliest decoded frame that moves: that frame needs a token from its future, so the gap is the decoder's algorithmic
latency, before and after the Gaussian smoothing applied at render time.

usage: python scripts/decoder_latency.py [tok_dir=outputs/tok-dense-rel-oct28] [data=outputs/run-full-dense-rel-oct28] [sigma=2.5]
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch
from vqvae_motion_tokenizer import smooth_features, VQVAETokenizer

tok = sys.argv[1] if len(sys.argv) > 1 else "outputs/tok-dense-rel-oct28"
data = sys.argv[2] if len(sys.argv) > 2 else "outputs/run-full-dense-rel-oct28"
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
vq = VQVAETokenizer(); vq.load(f"{tok}/tokenizer.pt"); vq.model.eval()
c = torch.as_tensor(np.load(f"{data}/val_motion.npy", mmap_mode="r")[0].astype(np.int64), device="cuda")[None]
with torch.no_grad():
    base = vq.model.decode_from_tokens(c)[0].permute(1, 0).float().cpu().numpy()
for s in (60, 75, 90):
    c2 = c.clone(); c2[0, s:] = (c2[0, s:] + 137) % 1024
    with torch.no_grad():
        d = vq.model.decode_from_tokens(c2)[0].permute(1, 0).float().cpu().numpy()
    fps = len(base) / c.shape[1] * 15
    first_slot_frame = s * len(base) / c.shape[1]
    for name, a, b in (("decoder", base, d), (f"smooth{sigma}", smooth_features(base, sigma), smooth_features(d, sigma))):
        moved = np.where(np.abs(a - b).max(1) > 1e-4)[0]
        print(f"DECLAT slot={s} {name} first_token_frame={first_slot_frame:.0f} earliest_moved={moved[0]} "
              f"reach_back={first_slot_frame - moved[0]:.0f} frames = {(first_slot_frame - moved[0]) / fps:.3f} s (fps {fps:.0f})", flush=True)
