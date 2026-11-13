"""Held-out SPEAKER test (scripts/heldout_spk.sh): speech match (per-window top-5 hit rate, as scripts/match_bootstrap.py,
retrieval within each eval group, averaged over draws) pooled over groups 1-7, split into the windows of the 60 held-out
people (outputs/heldout-spk-nov13/eval_windows.txt, val indices) and all other windows. Rows: the held-out-trained base
and UpBeat, and the full-data base and UpBeat on the same windows. Paired bootstrap (20000) over windows for the key gaps.

usage: python scripts/heldout_pool.py
"""
import os, sys, re, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from audio_motion_match import Match, motion_feats, crops, STARTS, NA, NM, A_FPS, FPS_M

F = "outputs/fix-oct21"
aud = np.load("outputs/run-full-q8-oct21/aligned_data/val_hubert.npy", mmap_mode="r")
stt = torch.load(f"{F}/match/match.pt", map_location="cpu")["state"]
mt = Match(stt["a_mean"], stt["a_std"], stt["m_mean"], stt["m_std"]).cuda(); mt.load_state_dict(stt); mt.eval()
HO = {int(x) for x in open("outputs/heldout-spk-nov13/eval_windows.txt").read().split()}
GA, GF = "sample1.3p1f15a1.5-5", "sample1.3p1f15g3.5"


@torch.no_grad()
def hits(sysn, prefix):
    out = {}
    for g in range(1, 8):
        ev, groups = f"{F}/e60_dt_v2_{sysn}_grp{g}", {}
        for p in glob.glob(f"{ev}/val*_{prefix}*.npy"):
            m = re.match(rf"val(\d+)_.*?_({re.escape(prefix)}(?:s\d+)?)\.npy$", os.path.basename(p))
            if m:
                groups.setdefault(m.group(2), {})[int(m.group(1))] = p
        assert len(groups) == 6, (ev, sorted(groups))
        for mode, files in sorted(groups.items()):
            idx = sorted(files)
            ea = mt.embed_audio(torch.as_tensor(np.stack([crops(np.asarray(aud[i], np.float32), STARTS, NA, A_FPS) for i in idx]),
                                                device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
            em = mt.embed_motion(torch.as_tensor(np.stack([crops(motion_feats(np.load(files[i])[:300]), STARTS, NM, FPS_M)
                                                           for i in idx]), device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
            cross = torch.einsum("nkd,mkd->knm", em, ea)
            own = cross.diagonal(dim1=1, dim2=2)
            h = ((cross > own[..., None]).sum(-1) < 5).float().mean(0).cpu().numpy()
            for i, v in zip(idx, h):
                out.setdefault(i, []).append(float(v))
    return {i: np.mean(v) for i, v in out.items()}


rows = {"held-out-trained base (g3.5)": hits("hobase", GF), "held-out-trained UpBeat (adaptive)": hits("hoUB", GA),
        "full-data base (g3.5)": hits("base", GF), "full-data UpBeat (adaptive)": hits("pas10ad", GA)}
allw = sorted(set.intersection(*[set(h) for h in rows.values()]))
ho, rest = [i for i in allw if i in HO], [i for i in allw if i not in HO]
print(f"windows: {len(allw)} total, {len(ho)} held-out speakers, {len(rest)} other")
for k, h in rows.items():
    print(f"POOL {k:36s} all {np.mean([h[i] for i in allw]):.3f}  heldout-spk {np.mean([h[i] for i in ho]):.3f}  "
          f"other {np.mean([h[i] for i in rest]):.3f}")


def boot(a, b, w, tag):
    d0 = np.array([rows[a][i] - rows[b][i] for i in w]); r = np.random.default_rng(0)
    d = np.array([d0[r.integers(0, len(d0), len(d0))].mean() for _ in range(20000)])
    print(f"BOOT {tag}: {a} - {b} = {d0.mean():+.3f} 95%CI=[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] "
          f"P(<=0)={(d <= 0).mean():.4f} n={len(w)}")


boot("held-out-trained UpBeat (adaptive)", "held-out-trained base (g3.5)", ho, "distillation gain, unseen speakers")
boot("held-out-trained UpBeat (adaptive)", "held-out-trained base (g3.5)", rest, "distillation gain, seen speakers")
boot("held-out-trained UpBeat (adaptive)", "full-data UpBeat (adaptive)", ho, "cost of never seeing the speaker")
boot("held-out-trained UpBeat (adaptive)", "full-data UpBeat (adaptive)", rest, "cost of 20% less data, seen speakers")
