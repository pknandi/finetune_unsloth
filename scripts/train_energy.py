"""Energy distillation (P1 stage B): a causal energy head on the online student that predicts the privileged teacher's
energy of a motion chunk from the past only. The teacher (offline model, full-window audio incl. the future) defines
E_T(chunk) = - mean guided log-probability of the chunk's tokens (scripts/energy_decode.teacher_energy). The head reads
the frozen student's causal frame states over the chunk (they depend on audio up to each frame and on the motion so
far, nothing later), pools them and outputs a scalar E_phi. Training: for each window, K student rollouts sampled with
the deployed decoding; for a random 1 s chunk the K teacher energies give a target ranking, and E_phi is trained with
a listwise loss (KL between softmax(-E_T / tau) and softmax(-E_phi)) plus a small regression term. At inference
(scripts/energy_decode.py energy "d") the student samples K candidate chunks and keeps the one with the lowest E_phi:
anticipating the teacher's future-aware judgement without hearing the future.

usage: python scripts/train_energy.py --student <ckpt> --teacher <ckpt> --s_data <dirs> --t_data <dirs> --out <dir>
       [--steps 2000] [--windows 8] [--K 4] [--tau 0.05] [--target lik|disc --energy_teacher <energy_teacher.pt>]
--target disc distils the trained privileged energy of scripts/train_energy_teacher.py instead of the teacher's
likelihood (which prefers safe low-motion chunks).
"""
import os, sys, json, time, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speech_to_motion_small import load_trained, T_M
from energy_decode import teacher_energy

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--s_data", required=True); ap.add_argument("--t_data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--windows", type=int, default=8)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--C", type=int, default=15)
ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--temp", type=float, default=1.3); ap.add_argument("--guid", type=float, default=3.5)
ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--target", default="lik", help="lik: teacher-likelihood energy; disc: the trained privileged energy (--energy_teacher)")
ap.add_argument("--energy_teacher", default="", help="scripts/train_energy_teacher.py output (energy_teacher.pt), for --target disc")
ap.add_argument("--slot", type=int, default=0, help="2026-11-11: >0 = per-slot head (zero-look-ahead decoding, energy_decode "
                "mode k1): the K candidates share one rollout and differ only in slot t; head and energy teacher read the "
                "states of the last SLOT slots ending at t (past + candidate)")
args, _ = ap.parse_known_args()
torch.manual_seed(0); np.random.seed(0)
os.makedirs(args.out, exist_ok=True)
student, teacher = load_trained(args.student)[0].eval(), load_trained(args.teacher)[0].eval()
for p in list(student.parameters()) + list(teacher.parameters()):
    p.requires_grad_(False)
Q, d = student.Q, student.d


from train_energy_head import EnergyHead  # noqa: E402  (shared with scripts/energy_decode.py)


head = EnergyHead(d).cuda()
ET = None
if args.target == "disc":
    _st = torch.load(args.energy_teacher, map_location="cpu")
    ET = EnergyHead(_st["d"]).cuda().eval(); ET.load_state_dict(_st["state"])
    for p in ET.parameters():
        p.requires_grad_(False)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)


def load(dirs, split):
    out = [(np.load(f"{x}/{split}_hubert.npy", mmap_mode="r"), np.load(f"{x}/{split}_motion.npy", mmap_mode="r"))
           for x in dirs.split(",")]
    return out


S_tr, T_tr = load(args.s_data, "train"), load(args.t_data, "train")
S_va, T_va = load(args.s_data.split(",")[0], "val"), load(args.t_data.split(",")[0], "val")   # val: first dirs only
# the streamed sets are row subsets of the dense sets: map rows by their window lists
import csv


def rows_map(sd, td, split):
    key = lambda w: (w["audio_filename"], w["start_sec"])
    tk = {key(w): i for i, w in enumerate(csv.DictReader(open(f"{td}/{split}_windows.csv")))}
    return np.array([tk[key(w)] for w in csv.DictReader(open(f"{sd}/{split}_windows.csv"))])


MAPS = [rows_map(s, t, "train") for s, t in zip(args.s_data.split(","), args.t_data.split(","))]
VMAP = rows_map(args.s_data.split(",")[0], args.t_data.split(",")[0], "val")


@torch.no_grad()
def rollouts(fs, real):
    """K deployed-decoding rollouts per window: (B*K, T_M, Q), seeded with the real first second"""
    B = len(fs)
    f = fs.repeat_interleave(args.K, 0)
    mem = student.encode_audio(f)
    mem = torch.cat([mem, student.a_null[None].to(mem.dtype).expand(len(f), -1, -1)])
    n = len(f)
    ids = torch.zeros(2 * n, T_M, Q, dtype=torch.long, device="cuda")
    ids[:, :15] = real[:, :15].clamp(min=0).repeat_interleave(args.K, 0).repeat(2, 1, 1)
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


def batch(S, T, maps, part, rows):
    rows = np.sort(rows)
    fs = torch.as_tensor(np.asarray(S[part][0][rows]), device="cuda")
    real = torch.as_tensor(np.asarray(S[part][1][rows]).astype(np.int64), device="cuda")
    ft = torch.as_tensor(np.asarray(T[part][0][maps[part][rows]]), device="cuda")
    return fs, ft, real


@torch.no_grad()
def branch_slot(fs, ids, t):
    """--slot: keep each window's first rollout, and resample slot t K times (deployed decoding) from it: (B*K, t+1, Q)"""
    B = len(fs)
    base = ids.view(B, args.K, T_M, Q)[:, 0].repeat_interleave(args.K, 0)[:, :t + 1].clone()
    f = fs.repeat_interleave(args.K, 0); mem = student.encode_audio(f); n = len(f)
    mem = torch.cat([mem, student.a_null[None].to(mem.dtype).expand(n, -1, -1)])
    cur = base.repeat(2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = student.frame_states(mem, cur)[:, t]
        cum = torch.zeros_like(h)
        for q in range(Q):
            lg = student.head(student.depth(h + cum + student.q_emb[q])).view(-1, Q, 1024)[:, q].float()
            lg = lg[n:] + args.guid * (lg[:n] - lg[n:])
            c = torch.multinomial(torch.softmax(lg / args.temp, -1), 1)[:, 0]
            cur[:n, t, q] = c; cur[n:, t, q] = c
            cum = cum + student.m_emb(c.repeat(2) + q * 1024)
    return cur[:n]


def step_loss(fs, ft, real, train=True):
    ids = rollouts(fs, real)                                                       # (B*K, T_M, Q)
    t = int(np.random.randint(15, T_M - args.C + 1)); e = t + args.C
    if args.slot:                                       # per-slot candidates, states over the last SLOT slots
        t = int(np.random.randint(15 + args.slot - 1, T_M - 1)); ids = branch_slot(fs, ids, t); e = t + 1
        t = e - args.slot
    with torch.no_grad():
        if args.target == "disc":                       # privileged discriminative energy on the teacher's states
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.slot:   # privileged teacher: window shifted by one state, so it includes the candidate's tokens
                    ix = torch.cat([ids[:, :e], torch.zeros_like(ids[:, :1])], 1)
                    ts = teacher.frame_states(teacher.encode_audio(ft.repeat_interleave(args.K, 0)), ix)[:, t + 1:e + 1].float()
                else:
                    ts = teacher.frame_states(teacher.encode_audio(ft.repeat_interleave(args.K, 0)), ids[:, :e])[:, t:e].float()
            et = ET(ts).view(len(fs), args.K)
        else:
            et = torch.cat([teacher_energy(teacher, ft[b], ids[b * args.K:(b + 1) * args.K, :e], t, e)
                            for b in range(len(fs))]).view(len(fs), args.K)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = student.frame_states(student.encode_audio(fs.repeat_interleave(args.K, 0)), ids[:, :e])[:, t:e].float()
    if args.slot:   # the candidate slot's tokens are not in any state (state s sees tokens < s): add their embeddings
        hs = hs.clone(); hs[:, -1] += sum(student.m_emb.weight[q * 1024 + ids[:, e - 1, q]] for q in range(Q)).float()
    ep = head(hs).view(len(fs), args.K)
    target = F.softmax(-et / args.tau, -1)
    lst = F.kl_div(F.log_softmax(-ep, -1), target, reduction="batchmean")
    z = lambda x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-4)
    reg = F.mse_loss(z(ep), z(et))
    acc = (ep.argmin(-1) == et.argmin(-1)).float().mean()                         # picks the teacher's best?
    gain = (et.gather(1, ep.argmin(-1, keepdim=True))[:, 0] - et.mean(-1)).mean()  # teacher energy of the pick vs random
    return lst + 0.1 * reg, acc, gain


if __name__ == "__main__":
    log = open(f"{args.out}/train_log.jsonl", "a")
    sizes = [len(m) for m in MAPS]
    t0, best = time.time(), -1e9
    for step in range(1, args.steps + 1):
        part = int(np.random.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
        fs, ft, real = batch(S_tr, T_tr, MAPS, part, np.random.choice(sizes[part], args.windows, replace=False))
        loss, acc, gain = step_loss(fs, ft, real)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"STEP {step} loss={loss.item():.4f} pick_teacher_best={acc.item():.2f} gain={gain.item():+.4f} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.val_every == 0:
            accs, gains = [], []
            rng = np.random.RandomState(0)
            with torch.no_grad():
                for k in range(8):
                    vr = rng.choice(len(VMAP), args.windows, replace=False)
                    vs, vt, vreal = batch(S_va, T_va, [VMAP], 0, vr)
                    _, a, g = step_loss(vs, vt, vreal, train=False)
                    accs.append(a.item()); gains.append(g.item())
            va, vg = float(np.mean(accs)), float(np.mean(gains))
            print(f"VAL step={step} pick_teacher_best={va:.3f} (chance {1 / args.K:.2f}) teacher_energy_gain={vg:+.4f}", flush=True)
            log.write(json.dumps({"step": step, "val_acc": va, "val_gain": vg}) + "\n"); log.flush()
            if -vg > best:
                best = -vg
                torch.save({"state": head.state_dict(), "d": d, "step": step, "C": args.C}, f"{args.out}/energy_head.pt")
    print("done", flush=True)
