"""Privileged discriminative energy E_T (P1, energy distillation, version 2). The teacher-likelihood energy prefers
safe low-motion chunks (oracle selection by it worsened BEAT2 FGD 0.545 -> 0.686), so the energy is instead trained,
as an energy-based model, to separate real motion from what the student produces: a head on the frozen OFFLINE
teacher's frame states over a 1 s chunk (computed with the full window's audio, i.e. it hears the future) outputs
E_T, trained with InfoNCE where the real chunk (teacher-forced real tokens, own audio) is the positive and the
negatives are (a) K student rollouts of the same window (deployed decoding) and (b) the real chunk of another window
under this window's audio (mismatched speech). Low energy = real-looking and in sync with the speech, including the
speech right after the chunk. scripts/train_energy.py --target disc then distils E_T into a causal head on the
student (no future), which scripts/energy_decode.py uses online.

usage: python scripts/train_energy_teacher.py --student <ckpt> --teacher <ckpt> --s_data <dirs> --t_data <dirs> --out <dir>
       [--steps 2000] [--windows 8] [--K 4]
"""
import os, sys, json, time, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--s_data", required=True); ap.add_argument("--t_data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--windows", type=int, default=8)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--C", type=int, default=15)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--temp", type=float, default=1.3)
ap.add_argument("--guid", type=float, default=3.5); ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--init", default="", help="start from another energy_teacher.pt (e.g. the Embody-3D one for BEAT2)")
args = ap.parse_args()
import train_energy as TE                                   # shares data loading and rollouts (its argparse is bypassed)
torch.manual_seed(0); np.random.seed(0)
os.makedirs(args.out, exist_ok=True)
from train_energy_head import EnergyHead
head = EnergyHead(TE.teacher.d).cuda()
if args.init:
    head.load_state_dict(torch.load(args.init, map_location="cpu")["state"])
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)


def tstates(ft, ids, t, e):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return TE.teacher.frame_states(TE.teacher.encode_audio(ft), ids[:, :e])[:, t:e].float()


def step_loss(fs, ft, real):
    B, K = len(fs), args.K
    ids = TE.rollouts(fs, real)                                                     # (B*K, T_M, Q)
    t = int(np.random.randint(15, ids.shape[1] - args.C + 1)); e = t + args.C
    realc = real.clamp(min=0)
    ok = (real[:, t:e] >= 0).all(-1).all(-1)                                        # windows with real tokens there
    pos = head(tstates(ft, realc, t, e))                                             # (B,)
    neg_s = head(tstates(ft.repeat_interleave(K, 0), ids, t, e)).view(B, K)          # student samples
    neg_a = head(tstates(ft, realc.roll(1, 0), t, e))                                # real motion, another window's audio
    logits = -torch.cat([pos[:, None], neg_s, neg_a[:, None]], 1)                    # low energy = high score
    loss = F.cross_entropy(logits[ok], torch.zeros(int(ok.sum()), dtype=torch.long, device="cuda"))
    acc = (logits[ok].argmax(-1) == 0).float().mean()
    return loss, acc


t0, best = time.time(), -1.0
log = open(f"{args.out}/train_log.jsonl", "a")
sizes = [len(m) for m in TE.MAPS]
for step in range(1, args.steps + 1):
    part = int(np.random.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
    fs, ft, real = TE.batch(TE.S_tr, TE.T_tr, TE.MAPS, part, np.random.choice(sizes[part], args.windows, replace=False))
    loss, acc = step_loss(fs, ft, real)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"STEP {step} loss={loss.item():.4f} real_ranked_first={acc.item():.2f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    if step % args.val_every == 0:
        rng = np.random.RandomState(0); accs = []
        with torch.no_grad():
            for k in range(8):
                vs, vt, vreal = TE.batch(TE.S_va, TE.T_va, [TE.VMAP], 0, rng.choice(len(TE.VMAP), args.windows, replace=False))
                accs.append(step_loss(vs, vt, vreal)[1].item())
        va = float(np.mean(accs))
        print(f"VAL step={step} real_ranked_first={va:.3f} (chance {1 / (args.K + 2):.2f})", flush=True)
        log.write(json.dumps({"step": step, "val_acc": va}) + "\n"); log.flush()
        if va > best:
            best = va
            torch.save({"state": head.state_dict(), "d": TE.teacher.d, "step": step, "C": args.C}, f"{args.out}/energy_teacher.pt")
print("done", flush=True)
