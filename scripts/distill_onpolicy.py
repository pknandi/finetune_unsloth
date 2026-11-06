"""On-policy future-aware distillation (2026-11-04). Student: the online-honest model (speech_to_motion_small.py,
--causal_audio, streamed HuBERT: hears no audio after its own frame). Teacher: the offline model (F2, full-window
HuBERT: hears the whole window incl. the future), frozen. Each step:
  1. the student samples its own motion for a batch of windows (temperature 1, no guidance; half the batches seeded
     with the real first 15 frames, as in eval);
  2. both models score every token of that rollout (teacher forcing on the student's own codes);
  3. loss = reverse KL(student || teacher) per frame and RVQ level (full distribution), on the sampled frames,
     + --nll_w x the usual NLL on the real motion (with --cond_drop, so guidance keeps working).
So the student learns what the future-aware teacher would do from the histories the student itself produces.
--kl_w 0 is the NLL-only control (same data, steps, schedule; no teacher). --fwd_w adds forward KL(teacher || student)
on the student's own rollouts (mass-covering: against reverse KL's diversity loss).
Checkpoints every --eval_every steps are scored on held-out val windows (not the 60 eval windows): val NLL and the
on-policy reverse KL to the teacher; best/ = lowest on-policy val KL.
Guided / deployment-matched variant (2026-11-06): --roll_temp / --roll_guid sample the rollouts with the deployed
decoding (e.g. 1.3 / 3.5, guidance on the student's own null-audio memory), --t_guid g makes the target the teacher's
classifier-free-guided distribution (null + g (audio - null), per level), so the student learns the guided policy
in one forward pass (run it at guidance 1). --sal_w a weights each token's KL by how much the audio moves the teacher
there (KL(teacher(audio) || teacher(null)), mean-normalised, ^a): distil where speech matters. --lvl_w w0,..,w{Q-1}
weights the RVQ levels (coarse = timing, fine = detail).
--anti_w a (contribution E, anticipatory encoder): a cosine loss making a linear map of the student's causal audio
memory at each slot predict the teacher's full-window audio memory at that slot (which has heard the future), so the
online encoder learns to anticipate what is coming instead of hearing it; the map is a training-only head (not saved).
--grpo_w adds GRPO (group-relative REINFORCE, --group rollouts per window) with a speech-match reward from a
separate reward matcher (scripts/audio_motion_match.py, MATCH_SEED=1 MATCH_POOL=1), next to the KL (RISE-style).

usage: python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best
         --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04
         --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir <dir> [--steps 3000]
"""
import os, sys, csv, json, time, math, random, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speech_to_motion_small import load_trained, save, motion_nll, T_M

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--s_data", required=True); ap.add_argument("--t_data", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--nll_w", type=float, default=0.5); ap.add_argument("--kl_w", type=float, default=1.0); ap.add_argument("--fwd_w", type=float, default=0.0); ap.add_argument("--hist_drop", type=float, default=0.2)
ap.add_argument("--cond_drop", type=float, default=0.1); ap.add_argument("--seed_frames", type=int, default=15)
ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--val_rows", type=int, default=64)
ap.add_argument("--tip_keep", type=float, default=1.0, help="train the KL only on this share of tokens, picked by TIP's Soft-OR score")
ap.add_argument("--t_temp", type=float, default=1.0, help="teacher temperature (> 1 softens the target)")
ap.add_argument("--grpo_w", type=float, default=0.0, help="weight of a GRPO term with a speech-match reward (0 = off)")
ap.add_argument("--group", type=int, default=4, help="GRPO: rollouts per window (the batch holds batch/group windows)")
ap.add_argument("--reward_ckpt", default="outputs/fix-oct21/match_reward/match.pt", help="GRPO reward matcher (not the eval one)")
ap.add_argument("--tok_dir", default="outputs/tok-dense-rel-oct28")
ap.add_argument("--roll_temp", type=float, default=1.0, help="rollout sampling temperature (deployed: 1.3)")
ap.add_argument("--roll_guid", type=float, default=1.0, help="rollout guidance on the student (deployed: 3.5)")
ap.add_argument("--t_guid", type=float, default=1.0, help="target = teacher's guided distribution at this scale")
ap.add_argument("--sal_w", type=float, default=0.0, help="weight tokens by (audio effect on the teacher)^sal_w")
ap.add_argument("--lvl_w", default="", help="comma list of Q per-level KL weights (default all 1)")
ap.add_argument("--anti_w", type=float, default=0.0, help="weight of the anticipation loss (student memory -> teacher memory)")
ap.add_argument("--seed", type=int, default=0, help="training seed (the eval60 hold-out stays seed 0)")
ap.add_argument("--keep_eval60", action="store_true", help="do not hold out the 60 Embody eval windows (BEAT2: own val split)")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)


def key(w):
    return w["audio_filename"], w["start_sec"]


def load_split(split):
    """student features (streamed), teacher features (full window), motion, for the same windows"""
    S, T, M = [], [], []
    for sd, td in zip(args.s_data.split(","), args.t_data.split(",")):
        if not os.path.exists(f"{sd}/{split}_hubert.npy"):
            continue
        src = {key(w): i for i, w in enumerate(csv.DictReader(open(f"{td}/{split}_windows.csv")))}
        rows = np.array([src[key(w)] for w in csv.DictReader(open(f"{sd}/{split}_windows.csv"))])
        S.append(np.load(f"{sd}/{split}_hubert.npy", mmap_mode="r"))
        T.append((np.load(f"{td}/{split}_hubert.npy", mmap_mode="r"), rows))
        M.append(np.load(f"{sd}/{split}_motion.npy"))
    return S, T, M


tr_S, tr_T, tr_M = load_split("train"); va_S, va_T, va_M = load_split("val")
sizes = [len(m) for m in tr_M]
print(f"train windows {sizes}, val {[len(m) for m in va_M]}", flush=True)
random.seed(0)
eval60 = set() if args.keep_eval60 else set(random.sample(range(len(va_M[0])), 60))   # scripts/aligned_eval.py's windows
va_rows = [i for i in np.random.RandomState(0).permutation(len(va_M[0])) if i not in eval60][:args.val_rows]
if args.seed:
    random.seed(args.seed)


def batch(S, T, M, part, rows):
    rows = np.sort(rows)
    fs = torch.as_tensor(np.asarray(S[part][rows]), device="cuda")
    ft = torch.as_tensor(np.asarray(T[part][0][T[part][1][rows]]), device="cuda")
    return fs, ft, torch.as_tensor(M[part][rows], device="cuda").long()


student = load_trained(args.student)[0]; teacher = load_trained(args.teacher)[0]
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
Q = student.Q
assert teacher.Q == Q and student.cfg["causal_audio"] and not teacher.cfg["causal_audio"]


@torch.no_grad()
def rollout(fs, real, P, temp=1.0, guid=1.0):
    """the student's own motion (B, T_M, Q) for audio fs, first P frames forced to the real codes; sampled at
    temperature temp with classifier-free guidance guid (as generate(): null + guid * (audio - null))"""
    was = student.training; student.eval()
    B, g2 = len(fs), guid != 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem = student.encode_audio(fs)
        if g2:
            mem = torch.cat([mem, student.a_null[None].to(mem.dtype).expand(B, -1, -1)])
        ids = torch.zeros(len(mem), T_M, Q, dtype=torch.long, device="cuda")
        ids[:, :P] = real[:, :P].clamp(min=0).repeat(2 if g2 else 1, 1, 1)
        for t in range(P, T_M):
            h = student.frame_states(mem, ids[:, :t + 1])[:, t]
            cum = torch.zeros_like(h)
            for q in range(Q):
                lg = student.head(student.depth(h + cum + student.q_emb[q])).view(-1, Q, 1024)[:, q].float()
                if g2:
                    lg = lg[B:] + guid * (lg[:B] - lg[B:])
                c = torch.multinomial(torch.softmax(lg / temp, -1), 1)[:, 0]
                if g2:
                    c = c.repeat(2)
                ids[:, t, q] = c
                cum = cum + student.m_emb(c + q * 1024)
        ids = ids[:B]
    student.train(was)
    return ids


if args.grpo_w:
    sys.path.insert(0, f"{REPO}/scripts")
    from vqvae_motion_tokenizer import VQVAETokenizer, Normalizer, features_to_smplx, smooth_features
    from audio_motion_match import Match, motion_feats, crops, STARTS, NA, NM, A_FPS, FPS_M
    vq = VQVAETokenizer(); vq.load(f"{args.tok_dir}/tokenizer.pt"); vq.model.eval()
    vq_norm = Normalizer(); vq_norm.load(f"{args.tok_dir}/normalizer.npz")
    _st = torch.load(args.reward_ckpt, map_location="cpu")["state"]
    rmatch = Match(_st["a_mean"], _st["a_std"], _st["m_mean"], _st["m_std"]).cuda(); rmatch.load_state_dict(_st); rmatch.eval()


@torch.no_grad()
def reward(ids, ft, G):
    """speech-match margin of each rollout (G rollouts per window, windows in order): cosine of its motion crops
    with its own audio minus the mean cosine with the other windows' audio at the same time (3 crops), from the
    reward matcher (trained with another seed than the eval matcher, on 25 fps audio repeated to 50 fps)"""
    feat = vq.model.decode_from_tokens(ids).permute(0, 2, 1).float().cpu().numpy() * vq_norm.std + vq_norm.mean
    mot = []
    for f in feat:
        sm = features_to_smplx(smooth_features(f, 2.5), vq_norm.orient_repr, vq_norm.pose_repr)
        mot.append(crops(motion_feats(sm[:300]), STARTS, NM, FPS_M))
    em = rmatch.embed_motion(torch.as_tensor(np.stack(mot), device="cuda").flatten(0, 1)).view(len(ids), len(STARTS), -1)
    a50 = ft[::G].float().repeat_interleave(2, 1).cpu().numpy()                 # one audio per window, 50 fps
    ea = rmatch.embed_audio(torch.as_tensor(np.stack([crops(a, STARTS, NA, A_FPS) for a in a50]), device="cuda")
                            .flatten(0, 1)).view(len(a50), len(STARTS), -1)
    sim = torch.einsum("rkd,wkd->rkw", em, ea).mean(1)                          # (rollouts, windows)
    own = torch.arange(len(ids), device="cuda") // G
    pos = sim.gather(1, own[:, None])[:, 0]
    return pos - (sim.sum(1) - pos) / (sim.shape[1] - 1)


LVL_W = (torch.tensor([float(x) for x in args.lvl_w.split(",")], device="cuda") if args.lvl_w else None)
assert LVL_W is None or len(LVL_W) == Q, f"--lvl_w needs {Q} values"


def rev_kl(fs, ft, ids, P, fwd=False, tip=1.0, t_temp=1.0, logp=False, t_guid=1.0, sal_w=0.0):
    """mean over sampled frames and levels of KL(student || teacher), full distributions; fwd=True also returns
    KL(teacher || student) on the same frames (mass-covering, as in GKD's mix), for --fwd_w.
    tip < 1: average only over the top tip share of (frame, level) tokens by TIP's Soft-OR score
    s = 1 - (1 - h)(1 - d) (Xu et al. 2026; h = student entropy, d = reverse KL, both min-max normalised):
    uncertain tokens and confident-but-wrong tokens. t_temp > 1 softens the teacher."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ls = student(fs, ids)
        with torch.no_grad():
            lt = teacher(ft, ids)
            ln = teacher(ft, ids, cond_drop=1.0) if (t_guid != 1.0 or sal_w) else None   # teacher on null audio
    lt = lt[:, P:].float()
    w = torch.ones(lt.shape[:-1], device="cuda")
    if ln is not None:
        ln = ln[:, P:].float()
        if sal_w:                                   # how much the audio moves the teacher at each (frame, level)
            pa, pn = F.log_softmax(lt, -1), F.log_softmax(ln, -1)
            sal = (pa.exp() * (pa - pn)).sum(-1)
            w = (sal / (sal.mean() + 1e-8)).clamp(min=0) ** sal_w
        if t_guid != 1.0:
            lt = ln + t_guid * (lt - ln)
    if LVL_W is not None:
        w = w * LVL_W
    ls, lt = F.log_softmax(ls[:, P:].float(), -1), F.log_softmax(lt / t_temp, -1)
    r, f = (ls.exp() * (ls - lt)).sum(-1), (lt.exp() * (lt - ls)).sum(-1)
    if tip < 1:
        with torch.no_grad():
            nrm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            sc = 1 - (1 - nrm(-(ls.exp() * ls).sum(-1))) * (1 - nrm(r))
            w = w * (sc >= sc.flatten().kthvalue(max(1, int(sc.numel() * (1 - tip)))).values).float()
    rev, fk = (r * w).sum() / w.sum(), (f * w).sum() / w.sum()
    if logp:     # per rollout: mean over frames of the summed log p of the sampled codes (for GRPO)
        lp = ls.gather(-1, ids[:, P:, :, None])[..., 0].sum(-1).mean(-1)
        return rev, fk, lp
    return (rev, fk) if fwd else rev


def validate():
    student.eval()
    kl, nll = [], []
    for i in range(0, len(va_rows), 16):
        fs, ft, m = batch(va_S, va_T, va_M, 0, va_rows[i:i + 16])
        ids = rollout(fs, m, args.seed_frames, args.roll_temp, args.roll_guid)
        with torch.no_grad():
            kl.append(rev_kl(fs, ft, ids, args.seed_frames, t_guid=args.t_guid).item())
            n, ok = motion_nll(student, fs, m)
            nll.append(((n * ok).sum() / ok.sum()).item())
    student.train()
    return float(np.mean(kl)), float(np.mean(nll))


anti = nn.Linear(student.d, teacher.d).cuda() if args.anti_w else None     # training-only anticipation head


def anti_loss(fs, ft):
    """1 - cos(anti(student causal memory), teacher full-window memory), mean over slots"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ms = student.encode_audio(fs)
        with torch.no_grad():
            mt = teacher.encode_audio(ft)
    return (1 - F.cosine_similarity(anti(ms.float()), mt.float(), dim=-1)).mean()


opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad] + (list(anti.parameters()) if anti else []),
                        lr=args.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                          0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
os.makedirs(args.out_dir, exist_ok=True)
log = open(f"{args.out_dir}/train_log.jsonl", "a")
kl0, nll0 = validate()
print(f"VAL step=0 onpolicy_kl={kl0:.4f} nll={nll0:.3f}", flush=True)
log.write(json.dumps({"step": 0, "val_kl": kl0, "val_nll": nll0}) + "\n"); log.flush()
best, t0, run = kl0, time.time(), []
save(f"{args.out_dir}/best", student, {"step": 0, "val_kl": kl0})
student.train()
for step in range(1, args.steps + 1):
    part = int(np.random.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
    G = args.group if args.grpo_w else 1
    fs, ft, m = batch(tr_S, tr_T, tr_M, part, np.random.choice(sizes[part], args.batch // G, replace=False))
    if G > 1:
        fs, ft, m = fs.repeat_interleave(G, 0), ft.repeat_interleave(G, 0), m.repeat_interleave(G, 0)
    P = args.seed_frames if np.random.rand() < 0.5 else 0
    fk, pg, rw = torch.zeros((), device="cuda"), torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
    if args.kl_w or args.fwd_w or args.grpo_w:
        ids = rollout(fs, m, P, args.roll_temp, args.roll_guid)
        kl, fk, lp = rev_kl(fs, ft, ids, P, fwd=True, tip=args.tip_keep, t_temp=args.t_temp, logp=True,
                            t_guid=args.t_guid, sal_w=args.sal_w)
        if args.grpo_w:                                         # group-relative advantage, one policy step per rollout
            r = reward(ids, ft, G); rw = r.mean()
            rg = r.view(-1, G); adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).flatten()
            pg = -(adv.detach() * lp).mean()
    else:                                                       # NLL-only control: same data, steps and schedule
        kl = torch.zeros((), device="cuda")
    n, ok = motion_nll(student, fs, m, args.hist_drop, args.cond_drop)
    loss = args.kl_w * kl + args.fwd_w * fk + args.grpo_w * pg + args.nll_w * (n * ok).sum() / ok.sum()
    if anti is not None:
        al = anti_loss(fs, ft); loss = loss + args.anti_w * al; rw = al.detach()     # logged in the reward= slot
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step(); sched.step()
    run.append((kl.item(), loss.item(), rw.item()))
    if step % 50 == 0:
        k, l, rr = np.mean(run, 0); run = []
        print(f"STEP {step} kl={k:.4f} loss={l:.4f} reward={rr:+.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        vk, vn = validate()
        print(f"VAL step={step} onpolicy_kl={vk:.4f} nll={vn:.3f}", flush=True)
        log.write(json.dumps({"step": step, "val_kl": vk, "val_nll": vn}) + "\n"); log.flush()
        save(f"{args.out_dir}/step{step}", student, {"step": step, "val_kl": vk})
        if vk < best:
            best = vk
            save(f"{args.out_dir}/best", student, {"step": step, "val_kl": vk})
print(f"done, best on-policy val KL {best:.4f} (start {kl0:.4f})", flush=True)
