"""Masked (non-autoregressive) speech-to-motion model, in the style of EMAGE / MoMask, on the same
data as speech_to_motion_small.py (HuBERT features + RVQ motion tokens).

Why: the autoregressive small model learns to lean on its own motion history; its audio use (val
NLL gap with audio rolled 5 s) peaks in the first epoch and fades as training goes on. Here a frame
never sees its own past tokens unless they are already decided: training hides a random share of
frames (cosine schedule, often nearly all) and predicts them from the audio plus the visible frames,
and generation starts from all frames hidden, i.e. from the audio alone, then fixes the most
confident frames over a few rounds.

Same audio encoder, level embeddings, depth head and --local_audio / --cond_drop options as
SmallS2M; the frame transformer is bidirectional and a frame's input is its own codes (visible) or a
mask embedding (hidden).

usage: python speech_to_motion_masked.py --data_dir <data> --out_dir <dir> [--epochs 60 --local_audio --cond_drop 0.1]
Each epoch prints: EVAL epoch= right= shift= other= gap_shift= ... (NLL of hidden frames at a fixed
50 % mask, right audio vs audio rolled 5 s vs another clip's audio). Saves <out_dir>/{best,last}/masked.pt.
"""
import os, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speech_to_motion_small import SmallS2M, Rows, T_M

SEED = 15          # frames of real motion given at generation with prefix (1 s); trained on half the rows


class MaskedS2M(SmallS2M):
    def frame_states(self, mem, codes, hidden):
        """codes (B, T, Q), hidden (B, T) bool -> (B, T, d); no causal mask"""
        B, T = codes.shape[:2]
        x = self.level_embed(codes).sum(2)
        x = torch.where(hidden[..., None], self.m_mask.to(x.dtype), x) + self.m_pos[:T]
        if self.local_audio:
            win = mem.unfold(1, self.L + 1, 1)[:, :T]
            x = x + self.a_local(win.transpose(2, 3).reshape(B, T, -1))
        return self.dec(x, mem)

    def forward(self, feats, codes, hidden, cond_drop=0.0):
        mem = self.encode_audio(feats)
        if cond_drop > 0:
            null = torch.rand(len(mem), 1, 1, device=mem.device) < cond_drop
            mem = torch.where(null, self.a_null.to(mem.dtype), mem)
        h = self.frame_states(mem, codes, hidden)
        return self.level_logits(h, self.level_embed(codes))


def random_hidden(codes, gen=None):
    """per row: hide a cos(pi/2 * u) share of frames (u ~ U(0,1)); half the rows keep the first SEED frames visible"""
    B, T = codes.shape[:2]
    u = torch.rand(B, 1, device=codes.device, generator=gen)
    ratio = torch.cos(math.pi / 2 * u).clamp(min=1.0 / T)
    hidden = torch.rand(B, T, device=codes.device, generator=gen) < ratio
    seeded = torch.rand(B, 1, device=codes.device, generator=gen) < 0.5
    hidden[:, :SEED] &= ~seeded
    return hidden & (codes >= 0).all(2)


def masked_nll(model, feats, codes, hidden, cond_drop=0.0):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(feats, codes, hidden, cond_drop)
    nll = F.cross_entropy(lg.float().reshape(-1, 1024), codes.clamp(min=0).reshape(-1).long(), reduction="none")
    return nll.view(codes.shape), hidden[..., None].expand_as(codes) & (codes >= 0)


def evaluate(model, feats, codes, idx, bs=32):
    """NLL of hidden frames at a fixed 50 % mask: right audio vs rolled 5 s vs another clip's audio"""
    model.eval()
    res = {k: [] for k in ["right", "shift", "other", "right0", "shift0", "other0"]}
    n = len(idx)
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    with torch.no_grad():
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            ob = [idx[(k + n // 2) % n] for k in range(i, min(i + bs, n))]
            c = torch.as_tensor(codes[b], device="cuda")
            f = torch.as_tensor(np.asarray(feats[b]), device="cuda")
            hidden = (torch.rand(c.shape[:2], device="cuda", generator=g) < 0.5) & (c >= 0).all(2)
            variants = {"right": f, "shift": torch.roll(f, f.shape[1] // 2, 1),
                        "other": torch.as_tensor(np.asarray(feats[ob]), device="cuda")}
            for k, fv in variants.items():
                nll, m = masked_nll(model, fv, c, hidden)
                m = m.float()
                res[k] += ((nll * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).tolist()
                res[k + "0"] += ((nll[..., 0] * m[..., 0]).sum(1) / m[..., 0].sum(1).clamp(min=1)).tolist()
    model.train()
    r = {k: float(np.mean(v)) for k, v in res.items()}
    r["win_shift"] = float(np.mean(np.array(res["shift"]) > np.array(res["right"])))
    return r


def save(out_dir, model, extra):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": model.cfg, **extra}, f"{out_dir}/masked.pt")


def load_trained(ckpt_dir):
    st = torch.load(f"{ckpt_dir}/masked.pt", map_location="cpu")
    model = MaskedS2M(**st["cfg"]).cuda()
    model.load_state_dict(st["state"])
    model.eval()
    return model, None, None


def _sample(lg, temperature, top_p):
    """lg (N, V) -> ids (N,), log-probs (N,) under the untruncated tempered distribution"""
    p = torch.softmax(lg / temperature, -1)
    sp, si = p.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    sp = sp * keep
    k = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
    tok = si.gather(-1, k)[:, 0]
    return tok, torch.log(p.gather(-1, tok[:, None])[:, 0] + 1e-12)


@torch.no_grad()
def generate(model, io, bos, feats, do_sample=True, temperature=1.0, top_p=1.0, temp_levels=0, prefix=None,
             guidance=1.0, steps=12):
    """feats: (F, 1024) HuBERT for the window + lookahead -> (T_M, Q) ids. Starts with every frame
    hidden (except prefix), samples all hidden frames each round, keeps the most confident, and
    re-hides the rest on a cosine schedule. Greedy (do_sample=False) takes argmax tokens."""
    Q = model.Q
    mem = model.encode_audio(torch.as_tensor(np.asarray(feats, dtype=np.float32), device="cuda")[None])
    if guidance != 1.0:
        mem = torch.cat([mem, model.a_null[None].to(mem.dtype)])
    G = len(mem)
    ids = torch.zeros(1, T_M, Q, dtype=torch.long, device="cuda")
    known = torch.zeros(T_M, dtype=torch.bool, device="cuda")
    if prefix is not None:
        ids[0, :len(prefix)] = torch.as_tensor(np.asarray(prefix, dtype=np.int64), device="cuda")
        known[:len(prefix)] = True
    n_free = int((~known).sum())
    for s in range(steps):
        hidden = ~known
        h = model.frame_states(mem, ids.expand(G, -1, -1), hidden[None].expand(G, -1))      # (G, T, d)
        cum = torch.zeros_like(h)
        new = torch.zeros(T_M, Q, dtype=torch.long, device="cuda")
        conf = torch.zeros(T_M, device="cuda")
        for q in range(Q):
            z = model.depth(h + cum + model.q_emb[q])
            lg = model.head(z).view(G, T_M, Q, 1024)[:, :, q].float()
            lg = lg[1] + guidance * (lg[0] - lg[1]) if G == 2 else lg[0]                     # (T, 1024)
            t = temperature if not temp_levels or q < temp_levels else 1.0
            if do_sample:
                tok, lp = _sample(lg, t, top_p)
            else:
                lp_all = torch.log_softmax(lg, -1); tok = lp_all.argmax(-1); lp = lp_all.max(-1).values
            new[:, q] = tok
            if q == 0:
                conf = lp
            cum = cum + model.m_emb.weight[q * 1024 + tok][None]
        # keep this many frames hidden after the round (cosine schedule, 0 after the last)
        n_hide = int(n_free * math.cos(math.pi / 2 * (s + 1) / steps)) if s < steps - 1 else 0
        cand = torch.where(hidden)[0]
        score = conf[cand]
        order = cand[score.argsort(descending=True)]
        fix = order[:max(len(cand) - n_hide, 0)]
        ids[0, fix] = new[fix]
        known[fix] = True
        if known.all():
            break
    return ids[0].cpu().numpy()


def train(args):
    torch.manual_seed(0); np.random.seed(0)
    meta = json.load(open(f"{args.data_dir}/meta.json"))
    Q, L = meta["num_quantizers"], meta["lookahead"]
    tr_m = np.load(f"{args.data_dir}/train_motion.npy"); va_m = np.load(f"{args.data_dir}/val_motion.npy")
    tr_f = np.load(f"{args.data_dir}/train_hubert.npy", mmap_mode="r")
    va_f = np.load(f"{args.data_dir}/val_hubert.npy", mmap_mode="r")
    if args.max_rows:
        tr_m, tr_f = tr_m[:args.max_rows], Rows(tr_f, np.arange(args.max_rows))
    samp = np.asarray(tr_f[np.linspace(0, len(tr_f) - 1, min(256, len(tr_f))).astype(int)], np.float32).reshape(-1, 1024)
    model = MaskedS2M(Q, L, args.d, args.enc_layers, args.dec_layers, samp.mean(0), samp.std(0) + 1e-4,
                      args.local_audio, args.audio_drop, args.cond_drop).cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    spe = math.ceil(len(tr_m) / args.batch)
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    va_idx = list(np.random.RandomState(0).permutation(len(va_m))[:args.val_rows])
    tr_idx_eval = list(np.random.RandomState(1).permutation(len(tr_m))[:min(256, len(tr_m))])
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(f"{args.out_dir}/train_log.jsonl", "a")
    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(tr_m))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = np.sort(perm[i:i + args.batch])
            c = torch.as_tensor(tr_m[b], device="cuda")
            f = torch.as_tensor(np.asarray(tr_f[b]), device="cuda")
            nll, m = masked_nll(model, f, c, random_hidden(c), args.cond_drop)
            loss = (nll * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        ev = evaluate(model, va_f, va_m, va_idx)
        tr = evaluate(model, tr_f, tr_m, tr_idx_eval)
        print(f"EVAL epoch={ep} right={ev['right']:.3f} shift={ev['shift']:.3f} other={ev['other']:.3f} "
              f"gap_shift={ev['shift'] - ev['right']:+.3f} gap_other={ev['other'] - ev['right']:+.3f} "
              f"gap0_shift={ev['shift0'] - ev['right0']:+.3f} win_shift={ev['win_shift']:.2f} "
              f"train_right={tr['right']:.3f} train_loss={tot / spe:.3f} {(time.time() - t0) / ep:.0f}s/epoch", flush=True)
        log.write(json.dumps({"epoch": ep, "val": ev, "train": tr}) + "\n"); log.flush()
        save(f"{args.out_dir}/last", model, {"epoch": ep})
        if ev["right"] < best:
            best, bad = ev["right"], 0
            save(f"{args.out_dir}/best", model, {"epoch": ep})
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"early stop: val NLL has not improved for {bad} epochs (best {best:.3f})", flush=True)
                break
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=4); ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--local_audio", action="store_true")
    ap.add_argument("--audio_drop", type=float, default=0.0)
    ap.add_argument("--cond_drop", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val_rows", type=int, default=531)
    ap.add_argument("--max_rows", type=int, default=0)
    train(ap.parse_args())
