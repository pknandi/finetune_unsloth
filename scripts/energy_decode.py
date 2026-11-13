"""Energy-guided streaming decoding (P1, after EBT-Policy's energy-based action selection). Generation proceeds in
chunks of C motion slots: K candidate continuations of the next chunk are sampled from the same history, each is
decoded (causal decoder, one-sided smoothing) and scored by an energy E = -cos(motion embedding, audio embedding) of
the last 3 s (45 slots) of motion ending at the chunk end, from the reward matcher (scripts/audio_motion_match.py,
MATCH_SEED=1 MATCH_POOL=1; not the eval matcher); the lowest-energy candidate is kept and generation continues.
energy="c" (causal): the audio crop ends at the chunk end and uses the streamed features (no future).
energy="o<d>" (oracle, analysis only): full-window features shifted d 50 fps frames into the future (people's motion
leads their speech by ~170 ms), i.e. what a model that could see a little ahead would prefer. Stage 2 trains a causal
energy model to predict the oracle energy from the past (anticipatory energy).
energy="T" (oracle, analysis): the trained privileged discriminative energy (scripts/train_energy_teacher.py, t_head=)
on the offline teacher's states with full-window audio.
energy="t" (oracle, analysis): the privileged teacher's energy, minus the mean guided log-probability of the chunk under
the offline model that hears the whole window (teacher= checkpoint, t_feats= its full-window features); no matcher.
"""
import os, sys
import numpy as np
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts")
from speech_to_motion_small import T_M
from audio_motion_match import Match, motion_feats, NM, NA
from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, features_to_smplx, smooth_features

_M = {}


def _matcher(path="outputs/fix-oct21/match_reward/match.pt"):
    if path not in _M:
        st = torch.load(path, map_location="cpu")["state"]
        m = Match(st["a_mean"], st["a_std"], st["m_mean"], st["m_std"]).cuda(); m.load_state_dict(st); m.eval()
        _M[path] = m
    return _M[path]


def _tok(tok_dir):
    if tok_dir not in _M:
        vq = VQVAETokenizer(); vq.load(f"{tok_dir}/tokenizer.pt"); vq.model.eval()
        nm = Normalizer(); nm.load(f"{tok_dir}/normalizer.npz")
        _M[tok_dir] = (vq, nm)
    return _M[tok_dir]


@torch.no_grad()
def _sample_chunk(model, mem, ids, t, C, K, temperature, guidance, adapt=None):
    """K continuations (K, C, Q) of ids[:t] (one window), sampled like generate() (top-p 1); adapt=(g_min, g_max):
    per frame and candidate, guidance from the normalised entropy of the conditional level-0 distribution (as
    generate(adapt=...))"""
    Q = model.Q
    g2 = guidance != 1.0 or adapt is not None
    m = mem.expand(K, -1, -1) if not g2 else torch.cat([mem[:1].expand(K, -1, -1), mem[1:].expand(K, -1, -1)])
    cur = ids[None].expand(len(m), -1, -1).clone()
    for s in range(t, min(t + C, T_M)):
        h = model.frame_states(m, cur[:, :s + 1])[:, s]
        cum = torch.zeros_like(h)
        g = guidance
        for q in range(Q):
            lg = model.head(model.depth(h + cum + model.q_emb[q])).view(-1, Q, 1024)[:, q].float()
            if adapt is not None and q == 0:
                pc = torch.softmax(lg[:K], -1)
                hn = -(pc * torch.log(pc + 1e-12)).sum(-1, keepdim=True) / np.log(1024)
                g = adapt[0] + (adapt[1] - adapt[0]) * hn                       # (K, 1)
            if g2:
                lg = lg[K:] + g * (lg[:K] - lg[K:])
            c = torch.multinomial(torch.softmax(lg / temperature, -1), 1)[:, 0]
            cur[:, s, q] = c.repeat(2) if g2 else c
            cum = cum + model.m_emb.weight[q * 1024 + cur[:, s, q]]
    return cur[:K, t:min(t + C, T_M)]


def _decode(ids, tok_dir, sigma):
    vq, nm = _tok(tok_dir)
    with torch.no_grad():
        f = vq.model.decode_from_tokens(ids.long()).permute(0, 2, 1).float().cpu().numpy() * nm.std + nm.mean
    return [features_to_smplx(smooth_features(x, sigma), nm.orient_repr, nm.pose_repr) for x in f]


_T = {}


def _teacher(path):
    if path not in _T:
        from speech_to_motion_small import load_trained
        _T[path] = load_trained(path)[0]
    return _T[path]


@torch.no_grad()
def teacher_energy(teacher, t_feats, cand_ids, t, e, g_t=3.5):
    """privileged energy of K candidate histories (K, e, Q): minus the mean log-probability of tokens t..e-1 under the
    offline teacher, which hears the whole window incl. the future (guided distribution, as the distillation target)"""
    f = (t_feats if torch.is_tensor(t_feats) else torch.as_tensor(np.asarray(t_feats), device="cuda"))[None].expand(len(cand_ids), -1, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        la = teacher(f, cand_ids).float()
        ln = teacher(f, cand_ids, cond_drop=1.0).float()
    lp = torch.log_softmax(ln + g_t * (la - ln), -1)[:, t:e]
    return -lp.gather(-1, cand_ids[:, t:e, :, None])[..., 0].mean((1, 2))


_H = {}


def _head(path):
    """the distilled causal energy head (scripts/train_energy.py)"""
    if path not in _H:
        from train_energy_head import EnergyHead
        st = torch.load(path, map_location="cpu")
        h = EnergyHead(st["d"]).cuda().eval(); h.load_state_dict(st["state"]); _H[path] = h
    return _H[path]


@torch.no_grad()
def generate_energy(model, feats, a_causal, a_full, tok_dir, K=4, C=15, energy="c", temperature=1.3, guidance=3.5,
                    prefix=None, sigma=-2.5, teacher=None, t_feats=None, head=None, retry=0.0, t_head=None, adapt=None,
                    win=None, plan=None):
    """feats: streamed features of the window (model input). a_causal / a_full: (frames, 1024) 50 fps audio for the
    energy (streamed 25 fps repeated x2 / full-window). Returns (T_M, Q) ids.
    2026-11-11: chunk selection has audio look-ahead: slot t is committed only after its chunk's last slot (t + C - 1)
    was sampled from audio up to that slot, so C = 15 gives (C - 1) / 15 s = 0.93 s. C = 1 (aligned_eval mode ...k1) is
    zero look-ahead (each slot committed from audio up to its own end). win (energy "d"): the head scores the states
    of the last `win` slots ending at the chunk end (past kept slots + the candidate) instead of the chunk only."""
    Q = model.Q
    mem = model.encode_audio(torch.as_tensor(np.asarray(feats), device="cuda")[None])
    if guidance != 1.0 or adapt is not None:
        mem = torch.cat([mem, model.a_null[None].to(mem.dtype)])
    ids = torch.zeros(T_M, Q, dtype=torch.long, device="cuda")
    t0 = 0
    if prefix is not None:
        t0 = len(prefix); ids[:t0] = torch.as_tensor(np.asarray(prefix, dtype=np.int64), device="cuda")
    shift = int(energy[1:]) if energy.startswith("o") else 0
    aud = a_full if energy.startswith("o") else a_causal
    mt = _matcher() if energy[0] in "co" else None
    tm = _teacher(teacher) if energy == "t" else None
    hd = _head(head) if energy == "d" else None
    tmT, hT = (_teacher(teacher), _head(t_head)) if energy == "T" else (None, None)
    seen = []                                                  # distilled energies of kept chunks (for the retry rule)
    if plan is not None and hd is not None:
        # 2026-11-12, zero look-ahead planning ("imagine, then commit"): at slot t, K rollouts of the next H slots are
        # sampled with the audio memory after slot t replaced by slot t's (nothing heard after t), scored by the head on
        # those imagined states, and only the first `commit` slots of the best rollout are kept; every kept slot is
        # thus chosen from audio up to t <= its own slot (zero look-ahead; slots t+1.. lag by up to commit-1 slots).
        H, cm = plan
        for t in range(t0, T_M, cm):
            mf = mem.clone(); mf[0, t + 1:] = mem[0, t]                     # conditional row only; null row unchanged
            cand = _sample_chunk(model, mf, ids, t, H, K, temperature, guidance, adapt)             # (K, h, Q)
            e = min(t + H, T_M)
            full = ids[None, :e].expand(K, -1, -1).clone(); full[:, t:e] = cand
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model.frame_states(mf[:1].expand(K, -1, -1), full)[:, t:e].float()
            k = int(hd(hs).argmin()); c = min(t + cm, T_M)
            ids[t:c] = full[k, t:c]
        return ids.cpu().numpy()
    for t in range(t0, T_M, C):
        cand = _sample_chunk(model, mem, ids, t, C, K, temperature, guidance, adapt)            # (K, c, Q)
        e = min(t + C, T_M)
        full = ids[None, :e].expand(K, -1, -1).clone(); full[:, t:e] = cand
        if energy == "r":                      # control (2026-11-10): same K candidates, picked at random, no energy
            ids[t:e] = cand[np.random.randint(K)]; continue
        if tm is not None:                                                             # privileged teacher energy
            ids[t:e] = cand[int(teacher_energy(tm, t_feats, full, t, e).argmin())]; continue
        if hT is not None:                                     # privileged discriminative energy (oracle, analysis)
            tf = torch.as_tensor(np.asarray(t_feats), device="cuda")[None].expand(K, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ts = tmT.frame_states(tmT.encode_audio(tf), full)[:, t:e].float()
            ids[t:e] = full[int(hT(ts).argmin()), t:e]; continue
        if hd is not None:                                     # distilled energy: the student's causal states only
            def score(c_ids):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hs = model.frame_states(mem[:1].expand(len(c_ids), -1, -1), c_ids)[:, (t if win is None else max(0, e - win)):e].float()
                if win is not None:
                    # 2026-11-12: the state at slot s is computed from tokens < s, so the candidate slot's own tokens
                    # never reach the states; add their embeddings to the last state (as the depth decoder's input),
                    # audio still only up to the candidate slot (heads trained with train_energy.py --slot)
                    hs = hs.clone(); hs[:, -1] += sum(model.m_emb.weight[q * 1024 + c_ids[:, e - 1, q]] for q in range(Q)).float()
                return hd(hs)
            en = score(full)
            # retry (EBT-Policy): if even the best candidate's energy is high relative to the chunks kept so far,
            # sample K more and keep the best of all 2K
            if retry and len(seen) >= 2 and float(en.min()) > np.mean(seen) + retry * (np.std(seen) + 1e-6):
                cand2 = _sample_chunk(model, mem, ids, t, C, K, temperature, guidance, adapt)
                full2 = ids[None, :e].expand(K, -1, -1).clone(); full2[:, t:e] = cand2
                full, en = torch.cat([full, full2]), torch.cat([en, score(full2)])
            k = int(en.argmin()); seen.append(float(en[k])); ids[t:e] = full[k, t:e]; continue
        if e < NM:                                                                     # not enough motion to score
            ids[t:e] = cand[0]; continue
        mot = np.stack([motion_feats(m)[e - NM:e] for m in _decode(full, tok_dir, sigma)])
        a0 = round((e - NM) * 50 / 15) + shift
        a = np.asarray(aud[max(0, a0):a0 + NA], np.float32)
        if len(a) < NA:
            a = np.concatenate([a, np.repeat(a[-1:], NA - len(a), 0)])
        em = mt.embed_motion(torch.as_tensor(mot, device="cuda"))
        ea = mt.embed_audio(torch.as_tensor(a[None], device="cuda"))
        en = -(em * ea).sum(-1)                                                       # lower = better fit
        ids[t:e] = cand[int(en.argmin())]
    return ids.cpu().numpy()
