"""Small from-scratch speech-to-motion baseline (no LLM), trained on the same data as
speech_to_motion_aligned.py (<data_dir> from its --build: HuBERT-large layer-12 features + RVQ motion
tokens), so the two are directly comparable and scripts/aligned_eval.py scores both.

Model: a bidirectional transformer encodes the whole window's audio (it is all known before
generation), pooled to one slot per 15 fps motion frame plus the lookahead. A causal transformer
decoder runs over motion frames: frame t's input is the sum of frame t-1's level embeddings (randomly
replaced by a mask with --hist_drop), and it cross-attends to the audio. Within a frame, level q is
predicted from the frame's hidden state plus the embeddings of levels < q (residual-VQ is coarse to
fine), like the LLM's interleaved layout.

usage: python speech_to_motion_small.py --data_dir <aligned_data> --out_dir <dir> [--epochs 100 --hist_drop 0.2]
Each epoch prints: EVAL epoch= right= shift= other= gap_shift= gap_other= gap0_shift= win_shift= train_right=
Saves <out_dir>/best/small.pt (by val right NLL) and <out_dir>/last/small.pt.
"""
import os, csv, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_M = 150


class SmallS2M(nn.Module):
    def __init__(self, Q, L, d=512, enc_layers=4, dec_layers=6, feat_mean=None, feat_std=None, local_audio=False, audio_drop=0.0, cond_drop=0.0, causal_audio=False, feat_shift=0, partner=0,
                 p_mean=None, p_std=None, any_shifts=None, onset=0, speakers=0, spk_default=0):
        super().__init__()
        self.Q, self.L, self.d, self.local_audio = Q, L, d, local_audio
        self.cfg = dict(Q=Q, L=L, d=d, enc_layers=enc_layers, dec_layers=dec_layers, local_audio=local_audio,
                        audio_drop=audio_drop, cond_drop=cond_drop, causal_audio=causal_audio, feat_shift=feat_shift,
                        partner=partner, any_shifts=any_shifts, onset=onset, speakers=speakers, spk_default=spk_default)
        # --speakers n (BEAT2 all-speaker training, 2026-11-08): a zero-initialised speaker embedding added to the audio
        # input, so training on every BEAT2 speaker does not average their styles; model.cur_spk (B,) holds the batch's
        # speaker numbers in training, otherwise every row uses spk_default (BEAT2 speaker 2 for our benchmark)
        self.speakers, self.spk_default, self.cur_spk = speakers, spk_default, None
        if speakers:
            self.spk_emb = nn.Embedding(speakers, d)
            nn.init.zeros_(self.spk_emb.weight)
        # --onset k (speech-onset anticipation, 2026-11-07): a head on each causal audio slot predicts k classes (speaking
        # now / next speech in 1..15 slots / none within 1 s; scripts/build_onset_labels.py, a training-only target) and
        # its predicted distribution is added back to that slot through a zero-initialised projection, so the decoder
        # gets an explicit "speech is about to start" signal computed from the past only
        self.onset = onset
        if onset:
            self.on_head = nn.Linear(d, onset)
            self.on_emb = nn.Linear(onset, d, bias=False)
            nn.init.zeros_(self.on_emb.weight)
        self.on_logits = None
        # --any_shifts "0,3,7,13" (one model, any latency; O1, 2026-11-06): each training batch draws one of these feature
        # shifts (look-ahead s / 25 s) and adds its learned latency embedding to the audio input; at inference the
        # latency is picked by setting model.feat_shift (load_trained: env ANY_SHIFT), so one model serves every latency
        self.any_shifts = list(any_shifts) if any_shifts else None
        if self.any_shifts:
            self.lat_emb = nn.Embedding(len(self.any_shifts), d)
            nn.init.zeros_(self.lat_emb.weight)
        # --partner (conversation, 2026-11-06): the other participants' causal log-mel (scripts/build_partner.py), packed
        # after the 1024 HuBERT dims of feats and added to the audio input through a zero-initialised projection, so a
        # checkpoint without it behaves exactly as before at init; it passes the same causal conv / mask as the audio
        self.partner = partner
        if partner:
            self.register_buffer("p_mean", torch.zeros(partner) if p_mean is None else torch.as_tensor(p_mean).float())
            self.register_buffer("p_std", torch.ones(partner) if p_std is None else torch.as_tensor(p_std).float())
            self.p_in = nn.Linear(partner, d)
            nn.init.zeros_(self.p_in.weight); nn.init.zeros_(self.p_in.bias)
        # --feat_shift s (with --causal_audio on streamed features): the 25 fps features are moved s frames earlier,
        # so frame t sees audio up to s / 25 s past its own slot: an online model with a fixed lookahead (latency)
        self.feat_shift = feat_shift
        # --causal_audio (strict online): frame t sees no audio after its own slot: causal conv and encoder,
        # local window t-L..t instead of t..t+L, cross-attention masked to slots <= t
        self.causal_audio = causal_audio
        self.audio_drop = nn.Dropout(audio_drop)     # on the encoded audio, against memorizing it
        self.register_buffer("feat_mean", torch.zeros(1024) if feat_mean is None else torch.as_tensor(feat_mean).float())
        self.register_buffer("feat_std", torch.ones(1024) if feat_std is None else torch.as_tensor(feat_std).float())
        self.a_in = nn.Linear(1024, d)
        self.a_conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.a_pos = nn.Parameter(torch.randn(T_M + L, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        self.m_emb = nn.Embedding(Q * 1024, d)
        self.m_start = nn.Parameter(torch.zeros(d))      # input for frame 0 (no previous frame)
        self.m_mask = nn.Parameter(torch.zeros(d))       # input for a dropped previous frame
        self.m_pos = nn.Parameter(torch.randn(T_M, d) * 0.02)
        # --local_audio: frame t's decoder input also gets the audio slots t..t+L (the frame and its
        # lookahead, as in the LLM layout), so timing is given directly instead of only via cross-attention
        self.a_local = nn.Linear((L + 1) * d, d) if local_audio else None
        dl = nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.depth = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.q_emb = nn.Parameter(torch.randn(Q, d) * 0.02)
        self.head = nn.Linear(d, Q * 1024)
        # --cond_drop: a learned "no audio" memory, swapped in for a random share of training rows, so
        # generate(guidance=g) can push the prediction away from the no-audio one (classifier-free guidance)
        self.a_null = nn.Parameter(torch.zeros(T_M + L, d)) if cond_drop > 0 else None
        self.register_buffer("causal", torch.triu(torch.full((T_M, T_M), float("-inf")), 1))

    def encode_audio(self, feats):
        """(B, F, 1024) -> (B, T_M + L, d)"""
        if self.__dict__.get("forecaster") is not None and self.feat_shift:
            return self._encode_forecast(feats)
        return self._encode(feats)

    @torch.no_grad()
    def _encode_forecast(self, feats, chunk=32):
        """imagined look-ahead (2026-11-08): zero real look-ahead for a model trained with feat_shift s. Memory slot t
        is computed from the real frames up to the slot's end e_t and, after them, the s frames that the speech
        forecaster (audio_forecaster.py) predicted at frame e_t; later frames are not used by the causal encoder."""
        B, Fr, _ = feats.shape; S = T_M + self.L; s = self.feat_shift
        feats = feats.float()
        fc = self.__dict__["forecaster"]
        if fc == "oracle":                       # control: the same per-slot path with the REAL future frames (leaks)
            idx = (torch.arange(Fr, device=feats.device)[:, None] + torch.arange(1, s + 1, device=feats.device)).clamp(max=Fr - 1)
            pred = feats[:, idx]
        else:
            pred = fc(feats)                                                   # (B, Fr, K, 1024)
        assert s <= pred.shape[2], (s, pred.shape)
        ends = [min(Fr, -(-(t + 1) * Fr // S)) - 1 for t in range(S)]           # last real frame pooled into slot t
        rows = []
        for c in range(0, S, chunk):
            ts = list(range(c, min(S, c + chunk)))
            f = feats.repeat(len(ts), 1, 1)                                     # (len(ts) * B, Fr, 1024), slot-major
            for j, t in enumerate(ts):
                e = ends[t]; n = min(s, Fr - 1 - e)
                if n > 0:
                    f[j * B:(j + 1) * B, e + 1:e + 1 + n] = pred[:, e, :n]
            m = self._encode(f)
            rows += [m[j * B:(j + 1) * B, t] for j, t in enumerate(ts)]
        return torch.stack(rows, 1)

    def _encode(self, feats):
        if self.feat_shift > 0:
            feats = torch.cat([feats[:, self.feat_shift:], feats[:, -1:].expand(-1, self.feat_shift, -1)], 1)
        elif self.feat_shift < 0:      # 2026-11-12: audio DELAYED by -feat_shift frames (predict-ahead training only)
            feats = torch.cat([feats[:, :1].expand(-1, -self.feat_shift, -1), feats[:, :self.feat_shift]], 1)
        if self.partner:
            feats, pf = feats[..., :1024], feats[..., 1024:]
        x = self.a_in((feats.float() - self.feat_mean) / self.feat_std)
        if self.partner:
            x = x + self.p_in((pf.float() - self.p_mean) / self.p_std)
        if self.any_shifts:
            x = x + self.lat_emb.weight[self.any_shifts.index(self.feat_shift)]
        if self.speakers:
            ids = self.cur_spk if self.cur_spk is not None else torch.full((len(x),), self.spk_default, device=x.device)
            x = x + self.spk_emb(ids.long())[:, None]
        x = x.transpose(1, 2)
        if self.causal_audio:
            x = F.pad(x, (2, 0))[:, :, :-2]      # delayed 2 frames: the centred kernel then sees t-4..t
        x = self.a_conv(x)
        x = F.adaptive_avg_pool1d(x, T_M + self.L).transpose(1, 2)
        mask = torch.triu(torch.full((T_M + self.L,) * 2, float("-inf"), device=x.device), 1) if self.causal_audio else None
        h = self.enc(x + self.a_pos, mask=mask)
        if self.onset:
            self.on_logits = self.on_head(h)                                    # (B, T_M + L, onset), kept for the loss
            h = h + self.on_emb(F.softmax(self.on_logits.float(), -1).to(h.dtype))
        return self.audio_drop(h)

    def level_embed(self, codes):
        """(B, T, Q) -> (B, T, Q, d)"""
        q = torch.arange(self.Q, device=codes.device)
        return self.m_emb(codes.clamp(min=0).long() + q * 1024)

    def frame_states(self, mem, codes, hist_drop=0.0):
        """Decoder hidden state per frame (B, T, d) from previous frames' codes."""
        B, T = codes.shape[:2]
        prev = self.level_embed(codes).sum(2)                                  # (B, T, d)
        if hist_drop > 0:
            drop = torch.rand(B, T, 1, device=codes.device) < hist_drop
            prev = torch.where(drop, self.m_mask.to(prev.dtype), prev)
        x = torch.cat([self.m_start.expand(B, 1, -1), prev[:, :-1]], 1) + self.m_pos[:T]
        if self.local_audio:
            m = F.pad(mem, (0, 0, self.L, 0)) if self.causal_audio else mem      # causal: slots t-L..t
            win = m.unfold(1, self.L + 1, 1)[:, :T]                           # (B, T, d, L + 1)
            x = x + self.a_local(win.transpose(2, 3).reshape(B, T, -1))
        mm = None
        if self.causal_audio:
            mm = torch.triu(torch.full((T, mem.shape[1]), float("-inf"), device=x.device), 1)
        return self.dec(x, mem, tgt_mask=self.causal[:T, :T], memory_mask=mm)

    def level_logits(self, h, lev):
        """h (B, T, d), lev (B, T, Q, d) embeddings of the frame's own codes -> (B, T, Q, 1024).
        Level q sees h + levels < q."""
        cum = torch.cumsum(lev, 2) - lev                                       # exclusive prefix sum
        z = self.depth(h[:, :, None] + cum + self.q_emb)
        lg = self.head(z).view(*z.shape[:3], self.Q, 1024)
        return lg[:, :, torch.arange(self.Q), torch.arange(self.Q)]          # level q's own slice

    def forward(self, feats, codes, hist_drop=0.0, cond_drop=0.0):
        mem = self.encode_audio(feats)
        if cond_drop > 0:
            null = torch.rand(len(mem), 1, 1, device=mem.device) < cond_drop
            mem = torch.where(null, self.a_null.to(mem.dtype), mem)
        h = self.frame_states(mem, codes, hist_drop)
        return self.level_logits(h, self.level_embed(codes))


def motion_nll(model, feats, codes, hist_drop=0.0, cond_drop=0.0):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(feats, codes, hist_drop, cond_drop)
    nll = F.cross_entropy(lg.float().reshape(-1, 1024), codes.clamp(min=0).reshape(-1).long(), reduction="none")
    return nll.view(codes.shape), (codes >= 0)


def evaluate(model, feats, codes, idx, bs=32):
    """Same three-way audio test as speech_to_motion_aligned.evaluate."""
    model.eval()
    res = {k: [] for k in ["right", "shift", "other", "right0", "shift0", "other0"]}
    n = len(idx)
    with torch.no_grad():
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            ob = [idx[(k + n // 2) % n] for k in range(i, min(i + bs, n))]
            c = torch.as_tensor(codes[b], device="cuda")
            f = torch.as_tensor(np.asarray(feats[b]), device="cuda")
            variants = {"right": f, "shift": torch.roll(f, f.shape[1] // 2, 1),
                        "other": torch.as_tensor(np.asarray(feats[ob]), device="cuda")}
            for k, fv in variants.items():
                nll, m = motion_nll(model, fv, c)
                m = m.float()
                res[k] += ((nll * m).sum((1, 2)) / m.sum((1, 2))).tolist()
                res[k + "0"] += ((nll[..., 0] * m[..., 0]).sum(1) / m[..., 0].sum(1)).tolist()
    model.train()
    r = {k: float(np.mean(v)) for k, v in res.items()}
    r["win_shift"] = float(np.mean(np.array(res["shift"]) > np.array(res["right"])))
    return r


def save(out_dir, model, extra):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": model.cfg, **extra}, f"{out_dir}/small.pt")


def load_trained(ckpt_dir):
    """Same return shape as speech_to_motion_aligned.load_trained: (model, io, bos)."""
    st = torch.load(f"{ckpt_dir}/small.pt", map_location="cpu")
    model = SmallS2M(**st["cfg"]).cuda()
    model.load_state_dict(st["state"])
    if model.any_shifts:                                   # any-latency model: which look-ahead to run at
        model.feat_shift = int(os.environ.get("ANY_SHIFT", "0"))
        assert model.feat_shift in model.any_shifts, (model.feat_shift, model.any_shifts)
    if os.environ.get("FORECAST") and model.feat_shift:   # imagined look-ahead: forecast the shifted-in frames
        from audio_forecaster import load_forecaster
        model.__dict__["forecaster"] = "oracle" if os.environ["FORECAST"] == "oracle" else load_forecaster(os.environ["FORECAST"])
    model.eval()
    return model, None, None


@torch.no_grad()
def generate(model, io, bos, feats, do_sample=False, temperature=1.0, top_p=0.9, temp_levels=0, prefix=None, guidance=1.0,
             adapt=None):
    """feats: (F, 1024) HuBERT for the window + lookahead. Returns (T_M, Q) ids.
    temp_levels > 0: temperature only on the first temp_levels RVQ levels (the rest at 1).
    prefix: (P, Q) ids forced as the first P frames (seed motion, e.g. the real first second).
    guidance g != 1 (model trained with --cond_drop): logits = no_audio + g * (audio - no_audio).
    adapt=(g_min, g_max) (P2, uncertainty-adaptive guidance): per frame, g = g_min + (g_max - g_min) * H, H the
    normalised entropy of the audio-conditioned level-0 distribution: strong guidance where the model is unsure."""
    Q = model.Q
    mem = model.encode_audio(torch.as_tensor(np.asarray(feats), device="cuda")[None])
    if guidance != 1.0 or adapt:
        mem = torch.cat([mem, model.a_null[None].to(mem.dtype)])
    ids = torch.zeros(1, T_M, Q, dtype=torch.long, device="cuda")

    def pick(lg, q):
        if not do_sample:
            return int(lg.argmax())
        p = torch.softmax(lg / (temperature if not temp_levels or q < temp_levels else 1.0), -1)
        sp, si = p.sort(descending=True)
        sp = sp * (sp.cumsum(0) - sp < top_p)
        return int(si[torch.multinomial(sp / sp.sum(), 1)])

    t0 = 0
    if prefix is not None:
        t0 = len(prefix)
        ids[0, :t0] = torch.as_tensor(np.asarray(prefix, dtype=np.int64), device="cuda")
    for t in range(t0, T_M):
        h = model.frame_states(mem, ids[:, :t + 1].expand(len(mem), -1, -1))[:, t]   # frame t sees frames < t
        cum = torch.zeros_like(h)
        g = guidance
        for q in range(Q):
            z = model.depth(h + cum + model.q_emb[q])
            lg = model.head(z).view(-1, Q, 1024)[:, q].float()
            if adapt and q == 0:                                       # this frame's guidance from its uncertainty
                pc = torch.softmax(lg[0], -1)
                hn = float(-(pc * torch.log(pc + 1e-12)).sum() / math.log(1024))
                g = adapt[0] + (adapt[1] - adapt[0]) * hn
            lg = lg[1] + g * (lg[0] - lg[1]) if len(lg) == 2 else lg[0]
            ids[0, t, q] = pick(lg, q)
            cum = cum + model.m_emb.weight[q * 1024 + ids[0, t, q]]
    return ids[0].cpu().numpy()


class Join:
    """HuBERT rows with the partner features appended on the last axis (zeros if partner is None), without copying"""
    def __init__(self, a, p, dim=80): self.a, self.p, self.dim = a, p, dim
    def __len__(self): return len(self.a)
    def __getitem__(self, b):
        h = np.asarray(self.a[b])
        p = np.zeros(h.shape[:-1] + (self.dim,), h.dtype) if self.p is None else np.asarray(self.p[b]).astype(h.dtype)
        return np.concatenate([h, p], -1)


class Rows:
    """row subset of a memmap without copying it"""
    def __init__(self, arr, idx): self.arr, self.idx = arr, np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, b): return self.arr[self.idx[b]]


class Cat:
    """two memmaps stacked along rows without copying; index with a sorted row array"""
    def __init__(self, a, b): self.a, self.b, self.n = a, b, len(a)
    def __len__(self): return self.n + len(self.b)
    def __getitem__(self, b):
        b = np.asarray(b); lo, hi = b[b < self.n], b[b >= self.n] - self.n
        return np.concatenate([np.asarray(self.a[lo]), np.asarray(self.b[hi])])


def train(args):
    torch.manual_seed(0); np.random.seed(0)
    meta = json.load(open(f"{args.data_dir}/meta.json"))
    Q, L = meta["num_quantizers"], meta["lookahead"]
    tr_m = np.load(f"{args.data_dir}/train_motion.npy"); va_m = np.load(f"{args.data_dir}/val_motion.npy")
    tr_f = np.load(f"{args.data_dir}/train_hubert.npy", mmap_mode="r")
    va_f = np.load(f"{args.data_dir}/val_hubert.npy", mmap_mode="r")
    if args.extra_data:
        # e.g. the windows the RMS gate left out (build_dense.py --gate lower), same tokenizer
        tr_m = np.concatenate([tr_m, np.load(f"{args.extra_data}/train_motion.npy")])
        tr_f = Cat(tr_f, np.load(f"{args.extra_data}/train_hubert.npy", mmap_mode="r"))
        print(f"extra_data {args.extra_data}: {len(tr_m)} train rows", flush=True)
    xidx = None
    if os.environ.get("EXCLUDE_SPEAKERS"):        # 2026-11-13 held-out speakers: drop every train window of these people
        ex = set(open(os.environ["EXCLUDE_SPEAKERS"]).read().split())
        names = [os.path.basename(os.path.dirname(os.path.dirname(w["audio_filename"])))
                 for d in [args.data_dir] + ([args.extra_data] if args.extra_data else [])
                 for w in csv.DictReader(open(f"{d}/train_windows.csv"))]
        assert len(names) == len(tr_m) and not args.partner and not args.max_rows and not args.speakers
        xidx = np.array([i for i, n in enumerate(names) if n not in ex])
        tr_m, tr_f = tr_m[xidx], Rows(tr_f, xidx)
        print(f"EXCLUDE_SPEAKERS {len(ex)} people: {len(names)} -> {len(xidx)} train rows", flush=True)
    p_stats = {}
    if args.partner:
        tr_p = np.load(f"{args.data_dir}/train_partner.npy", mmap_mode="r")
        if args.extra_data:
            tr_p = Cat(tr_p, np.load(f"{args.extra_data}/train_partner.npy", mmap_mode="r"))
        ps = np.asarray(tr_p[np.linspace(0, len(tr_m) - 1, min(256, len(tr_m))).astype(int)], np.float32).reshape(-1, args.partner)
        p_stats = dict(p_mean=ps.mean(0), p_std=ps.std(0) + 1e-4)
        tr_f = Join(tr_f, tr_p, args.partner)
        va_f = Join(va_f, np.load(f"{args.data_dir}/val_partner.npy", mmap_mode="r"), args.partner)
        print(f"partner features: {args.partner} dims", flush=True)
    if args.max_rows:
        tr_m, tr_f = tr_m[:args.max_rows], tr_f[:args.max_rows]
    if args.min_speech or args.speech_rep > 1:
        # share of the window where the person's own track is active (scripts/speech_activity.py)
        sp = np.load(f"{args.data_dir}/train_speech.npy")
        if args.extra_data:
            sp = np.concatenate([sp, np.load(f"{args.extra_data}/train_speech.npy")])
        sp = np.nan_to_num(sp, nan=0.0)
        if xidx is not None:
            sp = sp[xidx]
        assert len(sp) == len(tr_m), "speech filters need train_speech.npy for every row"
        keep = np.where(sp >= args.min_speech)[0]
        if args.speech_rep > 1:     # windows with speech share >= speech_thr appear speech_rep times per epoch
            keep = np.sort(np.concatenate([keep] + [keep[sp[keep] >= args.speech_thr]] * (args.speech_rep - 1)))
        print(f"min_speech {args.min_speech} speech_rep {args.speech_rep}@{args.speech_thr}: "
              f"{len(keep)} rows from {len(tr_m)}", flush=True)
        tr_m, tr_f = tr_m[keep], Rows(tr_f, keep)
    if args.min_change:
        # dense windows are calmer than val (one most-moving window per recording): keep the moving ones
        c0 = tr_m[:, :, 0].astype(np.int64); ok = (c0[:, 1:] >= 0) & (c0[:, :-1] >= 0)
        keep = np.where(((c0[:, 1:] != c0[:, :-1]) & ok).sum(1) / ok.sum(1).clip(1) >= args.min_change)[0]
        print(f"min_change {args.min_change}: {len(keep)}/{len(tr_m)} train rows", flush=True)
        tr_m, tr_f = tr_m[keep], Rows(tr_f, keep)
    samp = np.asarray(tr_f[np.linspace(0, len(tr_f) - 1, min(256, len(tr_f))).astype(int)], np.float32)[..., :1024].reshape(-1, 1024)
    model = SmallS2M(Q, L, args.d, args.enc_layers, args.dec_layers, samp.mean(0), samp.std(0) + 1e-4,
                     args.local_audio, args.audio_drop, args.cond_drop, args.causal_audio, args.feat_shift,
                     args.partner, **p_stats, any_shifts=[int(x) for x in args.any_shifts.split(",")] if args.any_shifts else None,
                     speakers=args.speakers, spk_default=args.spk_default).cuda()
    tr_spk = None
    if args.speakers:                                     # speaker number = first field of the BEAT2 recording id
        num = lambda a: int(os.path.basename(a).split("_")[0])
        tr_spk = np.array([num(w["audio_filename"]) for w in csv.DictReader(open(f"{args.data_dir}/train_windows.csv"))])
        assert not (args.extra_data or args.min_speech or args.speech_rep > 1 or args.min_change), "row filters not aligned with speakers"
        tr_spk = tr_spk[:args.max_rows] if args.max_rows else tr_spk
        assert len(tr_spk) == len(tr_m) and tr_spk.max() < args.speakers, (len(tr_spk), len(tr_m), tr_spk.max())
        print(f"speakers: {len(set(tr_spk.tolist()))} in train, default {args.spk_default}", flush=True)
    if args.init:
        st = torch.load(f"{args.init}/small.pt", map_location="cpu")["state"]     # incl. the init's HuBERT normaliser
        miss = model.load_state_dict(st, strict=False)
        assert not miss.unexpected_keys and set(miss.missing_keys) <= {"a_null", "p_in.weight", "p_in.bias", "p_mean", "p_std", "lat_emb.weight", "spk_emb.weight"}, miss
        print(f"init from {args.init}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    if args.partner and args.p_lr_mult != 1:            # the new partner projection learns faster than the init's weights
        pp = [p for n, p in model.named_parameters() if n.startswith("p_in.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("p_in.")]
        opt = torch.optim.AdamW([{"params": rest}, {"params": pp, "lr": args.lr * args.p_lr_mult}], lr=args.lr, weight_decay=0.05)
    else:
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
            if model.any_shifts:
                model.feat_shift = int(np.random.choice(model.any_shifts))
            if tr_spk is not None:
                model.cur_spk = torch.as_tensor(tr_spk[b], device="cuda")
            nll, m = motion_nll(model, f, c, args.hist_drop, args.cond_drop)
            model.cur_spk = None
            loss = (nll * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if model.any_shifts:                               # val NLL averaged over the latencies the model serves
            evs = []
            for sh in model.any_shifts:
                model.feat_shift = sh; evs.append(evaluate(model, va_f, va_m, va_idx))
            ev = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            print("EVAL_SHIFTS " + " ".join(f"s{sh}={e['right']:.3f}" for sh, e in zip(model.any_shifts, evs)), flush=True)
        else:
            ev = evaluate(model, va_f, va_m, va_idx)
        tr = evaluate(model, tr_f, tr_m, tr_idx_eval)
        print(f"EVAL epoch={ep} right={ev['right']:.3f} shift={ev['shift']:.3f} other={ev['other']:.3f} "
              f"gap_shift={ev['shift'] - ev['right']:+.3f} gap_other={ev['other'] - ev['right']:+.3f} "
              f"gap0_shift={ev['shift0'] - ev['right0']:+.3f} win_shift={ev['win_shift']:.2f} "
              f"train_right={tr['right']:.3f} train_loss={tot / spe:.3f} {(time.time() - t0) / ep:.0f}s/epoch", flush=True)
        log.write(json.dumps({"epoch": ep, "val": ev, "train": tr}) + "\n"); log.flush()
        save(f"{args.out_dir}/last", model, {"epoch": ep})
        if args.save_every and ep % args.save_every == 0:
            save(f"{args.out_dir}/ep{ep}", model, {"epoch": ep})
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=4); ap.add_argument("--dec_layers", type=int, default=6)
    ap.add_argument("--hist_drop", type=float, default=0.2)
    ap.add_argument("--local_audio", action="store_true", help="add frame-aligned audio to the decoder input")
    ap.add_argument("--causal_audio", action="store_true", help="strict online: no audio after frame t")
    ap.add_argument("--feat_shift", type=int, default=0, help="with --causal_audio: lookahead of this many 25 fps feature frames")
    ap.add_argument("--audio_drop", type=float, default=0.0, help="dropout on the encoded audio")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--speakers", type=int, default=0, help="speaker embedding table size (BEAT2: 31, ids = recording prefix)")
    ap.add_argument("--spk_default", type=int, default=0, help="speaker used outside training batches (BEAT2 benchmark: 2)")
    ap.add_argument("--save_every", type=int, default=0, help="also keep <out_dir>/ep<N> every N epochs (pick by val FGD)")
    ap.add_argument("--val_rows", type=int, default=531)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--min_change", type=float, default=0.0,
                    help="keep train rows whose level-0 token changes between frames at least this often")
    ap.add_argument("--cond_drop", type=float, default=0.0,
                    help="share of training rows with the audio replaced by a learned null (enables guidance)")
    ap.add_argument("--min_speech", type=float, default=0.0, help="drop train windows whose speech share is below this")
    ap.add_argument("--speech_rep", type=int, default=1, help="repeat windows with speech share >= --speech_thr this many times")
    ap.add_argument("--speech_thr", type=float, default=0.4)
    ap.add_argument("--extra_data", default="", help="dense dir whose train windows are appended (same tokenizer)")
    ap.add_argument("--init", default="", help="checkpoint dir (with small.pt) to start from")
    ap.add_argument("--any_shifts", default="", help="one model, any latency: comma list of feature shifts drawn per batch")
    ap.add_argument("--p_lr_mult", type=float, default=1.0, help="learning-rate multiplier for the partner projection")
    ap.add_argument("--partner", type=int, default=0, help="partner-audio dims (80: <data>/<split>_partner.npy from scripts/build_partner.py)")
    args = ap.parse_args()
    train(args)
