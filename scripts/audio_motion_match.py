"""Does generated motion follow its own speech? A small contrastive audio<->motion model, trained on
real training clips only, scores how well a 3 s motion crop fits the audio it was generated from.

  timing  three 3 s crops per clip (1.5-4.5, 4.25-7.25, 7-10 s): for each motion crop, is the audio
          crop from the SAME time in the same clip the best match? chance 0.33. Same speaker, same
          clip, so only the speech's timing/content can decide it: the main "follows the speech" test
  clip    for each motion crop, rank of its own audio among the same crop of all n clips (top-1,
          top-5; chance 1/n, 5/n). Can also be won by speaker identity or posture
  margin  mean cosine sim(own audio) - mean sim(other clips' audio at the same time)

Motion features: body + hand rotations (6D, global orientation excluded, so facing does not matter)
at 15 fps plus their frame differences. Audio: HuBERT-large layer 12 (50 fps) from <data_dir>.
Crops start at 1.5 s so a 1 s seed (aligned_eval ...f15 modes) is never scored.

usage: python scripts/audio_motion_match.py train <data_dir> <model_out.pt>
         (env MATCH_SEED=<int> for another split / init; MATCH_POOL=1 trains on 25 fps-pooled audio repeated to
          50 fps, the input a GRPO reward gets from the generator's own features)
       python scripts/audio_motion_match.py score <model.pt> <data_dir> <eval_dir> [<eval_dir> ...]
         (eval_dir: aligned_eval.py output, files val<i>_<subject>_<mode>.npy; also scores real motion
          as mode=gt, and other val clips' real motion as mode=other_gt)
Last lines: MATCH mode= timing= clip_top1= clip_top5= margin= n=
"""
import os, sys, re, csv, glob
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FPS_RAW, FPS_M, A_FPS = 30, 15, 50
WIN = 3.0
STARTS = (1.5, 4.25, 7.0)
NM, NA = int(WIN * FPS_M), int(WIN * A_FPS)          # 45 motion frames, 150 audio frames per crop
D_POSE = 51 * 6


def motion_feats(m):
    """(T, >=156) axis-angle SMPL-X at 30 fps -> (T/2, 2 * 306) pose 6D + frame differences, 15 fps"""
    from vqvae_motion_tokenizer import rotvec_to_6d
    p = rotvec_to_6d(m[::2, 3:156].reshape(-1, 3)).reshape(-1, D_POSE).astype(np.float32)
    return np.concatenate([p, np.diff(p, axis=0, prepend=p[:1])], 1)


def gt_window(audio, start, n=300):
    from vqvae_motion_tokenizer import load_smplx_sequence
    m = load_smplx_sequence(os.path.dirname(os.path.dirname(audio)))
    f0 = round(start * FPS_RAW)
    return m[f0:f0 + n]


def _load(row):
    try:
        m = gt_window(row[0], row[1])
        return motion_feats(m).astype(np.float16) if len(m) == 300 else None
    except Exception:
        return None


class Enc(nn.Module):
    def __init__(self, din, d=256, stride=1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, d), nn.GELU())
        self.conv = nn.Conv1d(d, d, 5, stride=stride, padding=2)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        x = self.conv(self.inp(x).transpose(1, 2)).transpose(1, 2)
        return F.normalize(self.out(self.tf(x).mean(1)), dim=-1)


class Match(nn.Module):
    def __init__(self, a_mean, a_std, m_mean, m_std):
        super().__init__()
        for k, v in dict(a_mean=a_mean, a_std=a_std, m_mean=m_mean, m_std=m_std).items():
            self.register_buffer(k, torch.as_tensor(v).float())
        self.a = Enc(1024, stride=3)
        self.m = Enc(2 * D_POSE)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def embed_audio(self, a):
        return self.a((a.float() - self.a_mean) / self.a_std)

    def embed_motion(self, m):
        return self.m((m.float() - self.m_mean) / self.m_std)


def crops(arr, starts, n, fps):
    return np.stack([arr[round(s * fps):round(s * fps) + n] for s in starts])


def train(data, out):
    tr = list(csv.DictReader(open(f"{data}/train_windows.csv")))
    cache = f"{os.path.dirname(out)}/match_train_motion.npy"
    if os.path.exists(cache):
        mot = np.load(cache)
    else:
        with ProcessPoolExecutor(32) as ex:
            res = list(ex.map(_load, [(r["audio_filename"], float(r["start_sec"])) for r in tr], chunksize=16))
        mot = np.stack([r if r is not None else np.full((150, 2 * D_POSE), np.nan, np.float16) for r in res])
        np.save(cache, mot)
    ok = np.where(~np.isnan(mot[:, 0, 0].astype(np.float32)))[0]
    print(f"{len(ok)}/{len(tr)} train clips with full motion", flush=True)
    aud = np.load(f"{data}/train_hubert.npy", mmap_mode="r")
    seed = int(os.environ.get("MATCH_SEED", 0))         # 1: the GRPO reward matcher, kept apart from the eval one (0)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    va = rng.choice(ok, 300, replace=False); trn = np.setdiff1d(ok, va)       # held-out train rows for early stopping
    samp = mot[trn[:500]].astype(np.float32).reshape(-1, 2 * D_POSE)
    asamp = np.asarray(aud[np.sort(trn[:200])], np.float32).reshape(-1, 1024)
    model = Match(asamp.mean(0), asamp.std(0) + 1e-4, samp.mean(0), samp.std(0) + 1e-4).cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    max_s = 10 - WIN

    pool = os.environ.get("MATCH_POOL") == "1"   # audio as the generator sees it: 25 fps pairs averaged, repeated to 50 fps

    def arow(r):
        if not pool:
            return aud[r]
        h = np.asarray(aud[r], np.float32); h = (h[0:len(h) // 2 * 2:2] + h[1:len(h) // 2 * 2:2]) / 2
        return np.repeat(h, 2, 0)

    def batch(rows, same_starts=None):
        a, m = [], []
        for r in rows:
            # two crops per clip, >= WIN apart: hard negatives that only timing can separate
            s1 = rng.uniform(0, max_s - WIN); s2 = rng.uniform(s1 + WIN, max_s)
            for s in (same_starts or (s1, s2)):
                k = round(s * FPS_M)
                m.append(mot[r, k:k + NM]); a.append(arow(r)[round(k * A_FPS / FPS_M):round(k * A_FPS / FPS_M) + NA])
        return (torch.as_tensor(np.stack(a).astype(np.float32), device="cuda"),
                torch.as_tensor(np.stack(m).astype(np.float32), device="cuda"))

    def val_acc():
        model.eval()
        with torch.no_grad():
            a, m = batch(np.sort(va), same_starts=(1.5, 4.25, 7.0))
            sim = model.embed_motion(m) @ model.embed_audio(a).T             # (3V, 3V)
            V = len(va); s = sim.view(V, 3, V, 3)
            timing = (s[torch.arange(V), :, torch.arange(V)].argmax(-1) == torch.arange(3, device="cuda")).float().mean()
            top1 = (sim.argmax(1) == torch.arange(3 * V, device="cuda")).float().mean()
        model.train()
        return timing.item(), top1.item()

    best, bad = -1, 0
    for ep in range(1, 101):
        perm = rng.permutation(trn)
        for i in range(0, len(perm) - 63, 64):
            a, m = batch(np.sort(perm[i:i + 64]))
            ea, em = model.embed_audio(a), model.embed_motion(m)
            lg = model.logit_scale.exp().clamp(max=100) * em @ ea.T
            y = torch.arange(len(lg), device="cuda")
            loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        t, c = val_acc()
        print(f"epoch {ep} loss {loss.item():.3f} heldout-train timing {t:.3f} top1 {c:.3f} (chance 0.333 / {1 / 900:.4f})", flush=True)
        if t > best:
            best, bad = t, 0
            torch.save({"state": model.state_dict()}, out)
        else:
            bad += 1
            if bad >= 8:
                break
    print(f"best heldout-train timing {best:.3f}", flush=True)


def score(ckpt, data, evs):
    st = torch.load(ckpt, map_location="cpu")["state"]
    model = Match(st["a_mean"], st["a_std"], st["m_mean"], st["m_std"]).cuda()
    model.load_state_dict(st); model.eval()
    va = list(csv.DictReader(open(f"{data}/val_windows.csv")))
    aud = np.load(f"{data}/val_hubert.npy", mmap_mode="r")
    subsets = []     # speaking / listening split, as in motion_dist_metrics.py
    if os.path.exists(f"{data}/val_speech.npy"):
        sp = np.load(f"{data}/val_speech.npy")
        subsets = [("speak", lambda i: sp[i] >= 0.4), ("listen", lambda i: not sp[i] >= 0.4)]
    for ev in evs:
        by_mode = defaultdict(dict)
        for p in glob.glob(f"{ev}/val*_*.npy"):
            mt = re.match(r"val(\d+)_(.+)_([a-z0-9_.]+?)\.npy$", os.path.basename(p))
            if mt and not p.endswith("_ids.npy"):
                by_mode[mt.group(3)][int(mt.group(1))] = p
        idx = sorted(set.intersection(*[set(d) for d in by_mode.values()]))
        gt = {i: gt_window(va[i]["audio_filename"], float(va[i]["start_sec"])) for i in idx}
        idx = [i for i in idx if len(gt[i]) == 300]
        others = [k for k in np.random.default_rng(1).permutation(len(va)) if k not in set(idx)][:len(idx)]
        sets = {"gt": [gt[i] for i in idx],
                "other_gt": [gt_window(va[k]["audio_filename"], float(va[k]["start_sec"])) for k in others]}
        for mode, d in sorted(by_mode.items()):
            sets[mode] = [np.load(d[i])[:300] for i in idx]
        with torch.no_grad():
            a = torch.as_tensor(np.stack([crops(np.asarray(aud[i], np.float32), STARTS, NA, A_FPS) for i in idx]), device="cuda")
            ea = model.embed_audio(a.flatten(0, 1)).view(len(idx), len(STARTS), -1)          # (n, 3, d)
            print(f"== {ev}", flush=True)
            for mode, ms in sets.items():
                if any(len(m) < 300 for m in ms):
                    print(f"MATCH mode={mode} skipped (short)"); continue
                mm = torch.as_tensor(np.stack([crops(motion_feats(m), STARTS, NM, FPS_M) for m in ms]), device="cuda")
                em = model.embed_motion(mm.flatten(0, 1)).view(len(idx), len(STARTS), -1)
                own = torch.einsum("nkd,njd->nkj", em, ea)                                     # same clip: motion k vs audio j
                timing = (own.argmax(-1) == torch.arange(len(STARTS), device="cuda")).float().mean().item()
                cross = torch.einsum("nkd,mkd->knm", em, ea)                                   # same time: clip n motion vs clip m audio
                rank = (cross > cross.diagonal(dim1=1, dim2=2)[..., None]).sum(-1)             # (3, n)
                n = len(idx)
                margin = (cross.diagonal(dim1=1, dim2=2).mean() - (cross.sum((1, 2)) - cross.diagonal(dim1=1, dim2=2).sum(-1)).mean() / (n * (n - 1))).item()
                print(f"MATCH mode={mode} timing={timing:.3f} clip_top1={(rank == 0).float().mean().item():.3f} "
                      f"clip_top5={(rank < 5).float().mean().item():.3f} margin={margin:+.3f} n={n}", flush=True)
                for sub, sel in subsets:     # ranks still among all n clips; averaged over the subset only
                    k = torch.as_tensor([j for j, i in enumerate(idx) if sel(i)], device="cuda")
                    print(f"MATCHSUB mode={mode} sub={sub} clip_top5={(rank[:, k] < 5).float().mean().item():.3f} "
                          f"n={len(k)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
