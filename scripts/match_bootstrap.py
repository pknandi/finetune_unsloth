"""Is the speech-match gap between two models real? Per-clip top-5 hit rate (as scripts/audio_motion_match.py: rank of
own audio among the n clips' audio, 3 crops) averaged over a model's draws, then a paired bootstrap over clips of the
difference A - B. Prints each model's mean (compare with its log) and the 95% interval. Eval dirs may be different window groups
(EVAL_GROUP): windows are paired by val index.

usage: python scripts/match_bootstrap.py <evalA_dir>[,<evalA2_dir>...] <evalB_dir>[,...] [mode_prefix=sample1.3p1f15g3.5] [mode_prefix_B]
"""
import os, sys, re, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from audio_motion_match import Match, motion_feats, crops, STARTS, NA, NM, A_FPS, FPS_M

aud = np.load("outputs/run-full-q8-oct21/aligned_data/val_hubert.npy", mmap_mode="r")
stt = torch.load("outputs/fix-oct21/match/match.pt", map_location="cpu")["state"]
mt = Match(stt["a_mean"], stt["a_std"], stt["m_mean"], stt["m_std"]).cuda(); mt.load_state_dict(stt); mt.eval()


@torch.no_grad()
def hits(dirs, prefix):
    """{val window index: [per-draw top-5 hit rate]}; dirs may be different window groups (scored per dir)"""
    out = {}
    for ev in dirs.split(","):
        groups = {}
        for p in glob.glob(f"{ev}/val*_{prefix}*.npy"):
            m = re.match(rf"val(\d+)_.*?_({re.escape(prefix)}(?:s\d+)?)\.npy$", os.path.basename(p))
            if m:
                groups.setdefault(m.group(2), {})[int(m.group(1))] = p
        for mode, files in sorted(groups.items()):
            idx = sorted(files)
            ea = mt.embed_audio(torch.as_tensor(np.stack([crops(np.asarray(aud[i], np.float32), STARTS, NA, A_FPS) for i in idx]),
                                                device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
            em = mt.embed_motion(torch.as_tensor(np.stack([crops(motion_feats(np.load(files[i])[:300]), STARTS, NM, FPS_M)
                                                           for i in idx]), device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
            cross = torch.einsum("nkd,mkd->knm", em, ea)
            own = cross.diagonal(dim1=1, dim2=2)
            h = ((cross > own[..., None]).sum(-1) < 5).float().mean(0).cpu().numpy()
            print(f"  {ev.split('/')[-1]} {mode} top5={h.mean():.3f} n={len(idx)}", flush=True)
            for i, v in zip(idx, h):
                out.setdefault(i, []).append(float(v))
    return out


pa = sys.argv[3] if len(sys.argv) > 3 else "sample1.3p1f15g3.5"
pb = sys.argv[4] if len(sys.argv) > 4 else pa
HA, HB = hits(sys.argv[1], pa), hits(sys.argv[2], pb)
common = sorted(set(HA) & set(HB))
assert common, "no windows in common"
a, b = np.array([np.mean(HA[i]) for i in common]), np.array([np.mean(HB[i]) for i in common])
A, B = [0] * int(np.mean([len(HA[i]) for i in common])), [0] * int(np.mean([len(HB[i]) for i in common]))
r = np.random.default_rng(0)
d = np.array([(a - b)[r.integers(0, len(a), len(a))].mean() for _ in range(20000)])
print(f"BOOT windows={len(common)} A={a.mean():.3f} ({len(A)} draws) B={b.mean():.3f} ({len(B)} draws) A-B={a.mean() - b.mean():+.3f} "
      f"95%CI=[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] P(A<=B)={(d <= 0).mean():.4f}", flush=True)
