"""Energy-guided streaming decoding (P1, after EBT-Policy's energy-based action selection). Generation proceeds in
chunks of C motion slots: K candidate continuations of the next chunk are sampled from the same history, each is
decoded (causal decoder, one-sided smoothing) and scored by an energy E = -cos(motion embedding, audio embedding) of
the last 3 s (45 slots) of motion ending at the chunk end, from the reward matcher (scripts/audio_motion_match.py,
MATCH_SEED=1 MATCH_POOL=1; not the eval matcher); the lowest-energy candidate is kept and generation continues.
energy="c" (causal): the audio crop ends at the chunk end and uses the streamed features (no future).
energy="o<d>" (oracle, analysis only): full-window features shifted d 50 fps frames into the future (people's motion
leads their speech by ~170 ms), i.e. what a model that could see a little ahead would prefer. Stage 2 trains a causal
energy model to predict the oracle energy from the past (anticipatory energy).
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
def _sample_chunk(model, mem, ids, t, C, K, temperature, guidance):
    """K continuations (K, C, Q) of ids[:t] (one window), sampled like generate() (top-p 1)"""
    Q = model.Q
    g2 = guidance != 1.0
    m = mem.expand(K, -1, -1) if not g2 else torch.cat([mem[:1].expand(K, -1, -1), mem[1:].expand(K, -1, -1)])
    cur = ids[None].expand(len(m), -1, -1).clone()
    for s in range(t, min(t + C, T_M)):
        h = model.frame_states(m, cur[:, :s + 1])[:, s]
        cum = torch.zeros_like(h)
        for q in range(Q):
            lg = model.head(model.depth(h + cum + model.q_emb[q])).view(-1, Q, 1024)[:, q].float()
            if g2:
                lg = lg[K:] + guidance * (lg[:K] - lg[K:])
            c = torch.multinomial(torch.softmax(lg / temperature, -1), 1)[:, 0]
            cur[:, s, q] = c.repeat(2) if g2 else c
            cum = cum + model.m_emb.weight[q * 1024 + cur[:, s, q]]
    return cur[:K, t:min(t + C, T_M)]


def _decode(ids, tok_dir, sigma):
    vq, nm = _tok(tok_dir)
    with torch.no_grad():
        f = vq.model.decode_from_tokens(ids.long()).permute(0, 2, 1).float().cpu().numpy() * nm.std + nm.mean
    return [features_to_smplx(smooth_features(x, sigma), nm.orient_repr, nm.pose_repr) for x in f]


@torch.no_grad()
def generate_energy(model, feats, a_causal, a_full, tok_dir, K=4, C=15, energy="c", temperature=1.3, guidance=3.5,
                    prefix=None, sigma=-2.5):
    """feats: streamed features of the window (model input). a_causal / a_full: (frames, 1024) 50 fps audio for the
    energy (streamed 25 fps repeated x2 / full-window). Returns (T_M, Q) ids."""
    Q = model.Q
    mem = model.encode_audio(torch.as_tensor(np.asarray(feats), device="cuda")[None])
    if guidance != 1.0:
        mem = torch.cat([mem, model.a_null[None].to(mem.dtype)])
    ids = torch.zeros(T_M, Q, dtype=torch.long, device="cuda")
    t0 = 0
    if prefix is not None:
        t0 = len(prefix); ids[:t0] = torch.as_tensor(np.asarray(prefix, dtype=np.int64), device="cuda")
    shift = int(energy[1:]) if energy.startswith("o") else 0
    aud = a_full if energy.startswith("o") else a_causal
    mt = _matcher()
    for t in range(t0, T_M, C):
        cand = _sample_chunk(model, mem, ids, t, C, K, temperature, guidance)            # (K, c, Q)
        e = min(t + C, T_M)
        if e < NM:                                                                     # not enough motion to score
            ids[t:e] = cand[0]; continue
        full = ids[None, :e].expand(K, -1, -1).clone(); full[:, t:e] = cand
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
