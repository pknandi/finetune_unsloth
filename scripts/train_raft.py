"""Energy distilled into the policy (2026-11-12), zero look-ahead by construction. Energy-guided DECODING needs
look-ahead (a chunk is chosen after its audio was heard; per-slot selection does not help), so the privileged energy
is moved into TRAINING instead (rejection-sampling fine-tuning, RAFT / expert iteration): per training window the
online student samples K rollouts with its deployed decoding; the privileged energy (scripts/train_energy_teacher.py:
a head on the offline teacher's states, which hear the whole window incl. the future) scores each rollout (sum over
1 s chunks); the student is trained by NLL on the lowest-energy rollout (with --cond_drop so guidance keeps working)
plus --real_w x NLL on the real motion. At inference it is a plain student (no energy, no extra cost).

usage: python scripts/train_raft.py --student <ckpt> --teacher <ckpt> --energy_teacher <energy_teacher.pt> \
         --s_data <dirs> --t_data <dirs> --out <dir> [--steps 1500] [--windows 8] [--K 4] [--lr 3e-5]
"""
import os, sys, json, time, argparse, csv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
from speech_to_motion_small import load_trained, save, motion_nll, T_M
from train_energy_head import EnergyHead

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--energy_teacher", required=True)
ap.add_argument("--s_data", required=True); ap.add_argument("--t_data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--windows", type=int, default=8)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--temp", type=float, default=1.3); ap.add_argument("--guid", type=float, default=3.5)
ap.add_argument("--cond_drop", type=float, default=0.1); ap.add_argument("--real_w", type=float, default=0.5)
ap.add_argument("--val_every", type=int, default=100); ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dpo", type=float, default=0.0, help="2026-11-12: >0 = DPO with this beta (best vs worst rollout, frozen reference = the initial student) instead of NLL on the best (which collapsed: speech 0.11 at step 500)")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
student = load_trained(args.student)[0].cuda().train()
teacher = load_trained(args.teacher)[0].cuda().eval()
for p in teacher.parameters():
    p.requires_grad_(False)
_st = torch.load(args.energy_teacher, map_location="cpu")
ET = EnergyHead(_st["d"]).cuda().eval(); ET.load_state_dict(_st["state"])
for p in ET.parameters():
    p.requires_grad_(False)
Q = student.Q
ref = None
if args.dpo:
    ref = load_trained(args.student)[0].cuda().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05)


def load(dirs, split):
    return [(np.load(f"{x}/{split}_hubert.npy", mmap_mode="r"), np.load(f"{x}/{split}_motion.npy", mmap_mode="r"))
            for x in dirs.split(",")]


def rows_map(sd, td, split):
    key = lambda w: (w["audio_filename"], w["start_sec"])
    tk = {key(w): i for i, w in enumerate(csv.DictReader(open(f"{td}/{split}_windows.csv")))}
    return np.array([tk[key(w)] for w in csv.DictReader(open(f"{sd}/{split}_windows.csv"))])


S_tr, T_tr = load(args.s_data, "train"), load(args.t_data, "train")
S_va, T_va = load(args.s_data.split(",")[0], "val"), load(args.t_data.split(",")[0], "val")
MAPS = [rows_map(s, t, "train") for s, t in zip(args.s_data.split(","), args.t_data.split(","))]
VMAP = rows_map(args.s_data.split(",")[0], args.t_data.split(",")[0], "val")


def batch(S, T, maps, part, rows):
    rows = np.sort(rows)
    fs = torch.as_tensor(np.asarray(S[part][0][rows]), device="cuda")
    real = torch.as_tensor(np.asarray(S[part][1][rows]).astype(np.int64), device="cuda")
    ft = torch.as_tensor(np.asarray(T[part][0][maps[part][rows]]), device="cuda")
    return fs, ft, real


@torch.no_grad()
def rollouts(fs, real, K):
    """K deployed-decoding rollouts per window (B*K, T_M, Q), seeded with the real first second"""
    f = fs.repeat_interleave(K, 0)
    mem = student.encode_audio(f)
    mem = torch.cat([mem, student.a_null[None].to(mem.dtype).expand(len(f), -1, -1)])
    n = len(f)
    ids = torch.zeros(2 * n, T_M, Q, dtype=torch.long, device="cuda")
    ids[:, :15] = real[:, :15].clamp(min=0).repeat_interleave(K, 0).repeat(2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(15, T_M):
            h = student.frame_states(mem, ids[:, :t + 1])[:, t]
            cum = torch.zeros_like(h)
            for q in range(Q):
                lg = student.head(student.depth(h + cum + student.q_emb[q])).view(-1, Q, 1024)[:, q].float()
                lg = lg[n:] + args.guid * (lg[:n] - lg[n:])
                c = torch.multinomial(torch.softmax(lg / args.temp, -1), 1)[:, 0].repeat(2)
                ids[:, t, q] = c
                cum = cum + student.m_emb(c + q * 1024)
    return ids[:n]


@torch.no_grad()
def energy(ft, ids, K):
    """privileged energy of each rollout: sum of the energy-teacher head over its 1 s chunks after the seed"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ts = teacher.frame_states(teacher.encode_audio(ft.repeat_interleave(K, 0)), ids).float()
    return sum(ET(ts[:, t:t + 15]) for t in range(15, T_M - 14, 15)).view(-1, K)


if __name__ == "__main__":
    log = open(f"{args.out}/train_log.jsonl", "a")
    sizes = [len(m) for m in MAPS]; t0 = time.time()
    for step in range(1, args.steps + 1):
        part = int(np.random.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
        fs, ft, real = batch(S_tr, T_tr, MAPS, part, np.random.choice(sizes[part], args.windows, replace=False))
        student.eval(); ids = rollouts(fs, real, args.K); en = energy(ft, ids, args.K); student.train()
        best = ids.view(len(fs), args.K, T_M, Q)[torch.arange(len(fs)), en.argmin(-1)]          # (B, T_M, Q)
        n_r, ok_r = motion_nll(student, fs, real, cond_drop=args.cond_drop)
        lr_ = (n_r * ok_r).sum() / ok_r.sum()
        if args.dpo:          # preference: best vs worst rollout by the privileged energy, relative to the frozen start
            worst = ids.view(len(fs), args.K, T_M, Q)[torch.arange(len(fs)), en.argmax(-1)]
            lp = lambda m, x: -motion_nll(m, fs, x)[0][:, 15:].float().sum((1, 2))          # sequence log-prob (conditional)
            with torch.no_grad():
                rb, rw = lp(ref, best), lp(ref, worst)
            margin = (lp(student, best) - rb) - (lp(student, worst) - rw)
            lb = -torch.nn.functional.logsigmoid(args.dpo * margin).mean()
            msg = f"dpo={lb.item():.4f} margin={margin.mean().item():+.2f}"
        else:
            n_b, ok_b = motion_nll(student, fs, best, cond_drop=args.cond_drop)
            lb = (n_b[:, 15:] * ok_b[:, 15:]).sum() / ok_b[:, 15:].sum()
            msg = f"nll_best={lb.item():.3f}"
        loss = lb + args.real_w * lr_
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step % 25 == 0:
            print(f"STEP {step} {msg} nll_real={lr_.item():.3f} energy_best={en.min(-1).values.mean().item():.3f} "
                  f"energy_mean={en.mean().item():.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.val_every == 0:          # privileged energy of ONE deployed rollout on fixed val windows (lower = better)
            student.eval(); rng = np.random.RandomState(0); es = []
            for k in range(4):
                vs, vt, vr = batch(S_va, T_va, [VMAP], 0, rng.choice(len(VMAP), args.windows, replace=False))
                es.append(energy(vt, rollouts(vs, vr, 1), 1).mean().item())
            student.train()
            print(f"VAL step={step} energy_one_rollout={np.mean(es):.4f}", flush=True)
            log.write(json.dumps({"step": step, "val_energy": float(np.mean(es))}) + "\n"); log.flush()
        if step % args.save_every == 0:
            save(f"{args.out}/step{step}", student.eval(), {"step": step}); student.train()
    print("done", flush=True)
