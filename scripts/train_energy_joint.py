"""Joint energy distillation on Embody-3D + BEAT2 (2026-11-07). The BEAT2-only privileged energy was weak (BEAT2 speaker 2
is small: val 0.31 vs chance 0.17) and its distilled head did not help the BEAT2 test (FGD 0.614 vs 0.545), so one
privileged energy E_T and one causal head E_phi are trained on both datasets at once: each step draws a batch from
Embody-3D or BEAT2 (probability --mix for BEAT2), with that dataset's own student, offline teacher, data and deployed
decoding (scripts/train_energy.py is loaded once per dataset). The losses are those of scripts/train_energy_teacher.py
(--stage teacher: InfoNCE, real chunk vs student rollouts vs mismatched audio) and scripts/train_energy.py --target disc
(--stage head: listwise KL to the joint E_T on the student's causal states). Validation reports both datasets; the best
checkpoint is chosen on their mean. Output files have the same format as the single-dataset scripts, so
scripts/energy_decode.py / beat2_generate.py use them unchanged.

usage: python scripts/train_energy_joint.py --stage teacher|head --out <dir> [--steps 3000] [--mix 0.5]
       [--init <energy_teacher.pt>] [--energy_teacher <joint energy_teacher.pt> (stage head)] [--windows 8] [--val_every 250]
"""
import os, sys, json, time, argparse, importlib.util
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["teacher", "head"]); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--mix", type=float, default=0.5)
ap.add_argument("--windows", type=int, default=8); ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--val_batches", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--init", default=""); ap.add_argument("--energy_teacher", default="")
ap.add_argument("--emb_student", default="outputs/dt-Db-nov06/step3000")
ap.add_argument("--b2_student", default="outputs/beat2-dist-fwd-nov05/best")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
EMB = ["--student", args.emb_student, "--teacher", "outputs/dt-F2-cfg-oct30/best",
       "--s_data", "outputs/stream-f24-nov04,outputs/stream-lower24-nov04",
       "--t_data", "outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30", "--guid", "3.5", "--temp", "1.3"]
B2 = ["--student", args.b2_student, "--teacher", "outputs/beat2-ft-f2hw1.0-nov01/best",
      "--s_data", "outputs/beat2-stream-nov04", "--t_data", "outputs/beat2-spk2-oct30/rel-val-hw1.0", "--guid", "1", "--temp", "1.0"]
common = ["--out", args.out, "--windows", str(args.windows), "--tau", str(args.tau)]
common += ["--target", "disc", "--energy_teacher", args.energy_teacher] if args.stage == "head" else ["--target", "lik"]


def load_te(name, extra):
    """a separate copy of scripts/train_energy.py (its own student, teacher, data) for one dataset"""
    argv = sys.argv
    sys.argv = ["train_energy.py"] + extra + common
    spec = importlib.util.spec_from_file_location(name, f"{REPO}/scripts/train_energy.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.argv = argv
    return m


DS = {"embody": load_te("te_embody", EMB), "beat2": load_te("te_beat2", B2)}
torch.manual_seed(0); np.random.seed(0)
from train_energy_head import EnergyHead
d = DS["embody"].teacher.d if args.stage == "teacher" else DS["embody"].d
assert all((m.teacher.d if args.stage == "teacher" else m.d) == d for m in DS.values())
head = EnergyHead(d).cuda()
if args.init:
    head.load_state_dict(torch.load(args.init, map_location="cpu")["state"])
for m in DS.values():
    m.head = head                                      # one shared head; stage head also shares the joint E_T (m.ET)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)


def teacher_loss(m, fs, ft, real):
    """scripts/train_energy_teacher.py step_loss for dataset module m"""
    B, K, C = len(fs), m.args.K, m.args.C
    ids = m.rollouts(fs, real)
    t = int(np.random.randint(15, ids.shape[1] - C + 1)); e = t + C
    realc = real.clamp(min=0); ok = (real[:, t:e] >= 0).all(-1).all(-1)

    def ts(f, i):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m.teacher.frame_states(m.teacher.encode_audio(f), i[:, :e])[:, t:e].float()
    pos = head(ts(ft, realc))
    neg_s = head(ts(ft.repeat_interleave(K, 0), ids)).view(B, K)
    neg_a = head(ts(ft, realc.roll(1, 0)))
    logits = -torch.cat([pos[:, None], neg_s, neg_a[:, None]], 1)
    loss = F.cross_entropy(logits[ok], torch.zeros(int(ok.sum()), dtype=torch.long, device="cuda"))
    return loss, (logits[ok].argmax(-1) == 0).float().mean(), torch.zeros(())


def loss_fn(m, fs, ft, real):
    return teacher_loss(m, fs, ft, real) if args.stage == "teacher" else m.step_loss(fs, ft, real)


def draw(m, val, rng=np.random):
    if val:
        return m.batch(m.S_va, m.T_va, [m.VMAP], 0, rng.choice(len(m.VMAP), args.windows, replace=False))
    sizes = [len(x) for x in m.MAPS]
    part = int(np.random.choice(len(sizes), p=np.array(sizes) / sum(sizes)))
    return m.batch(m.S_tr, m.T_tr, m.MAPS, part, np.random.choice(sizes[part], args.windows, replace=False))


t0, best = time.time(), -1e9
log = open(f"{args.out}/train_log.jsonl", "a")
for step in range(1, args.steps + 1):
    name = "beat2" if np.random.rand() < args.mix else "embody"
    loss, acc, gain = loss_fn(DS[name], *draw(DS[name], False))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"STEP {step} {name} loss={loss.item():.4f} acc={acc.item():.2f} gain={gain.item():+.4f} "
              f"{(time.time() - t0) / step:.2f}s/step", flush=True)
    if step % args.val_every == 0:
        res = {}
        with torch.no_grad():
            for name, m in DS.items():
                rng = np.random.RandomState(0)
                r = [loss_fn(m, *draw(m, True, rng)) for _ in range(args.val_batches)]
                res[name] = (float(np.mean([x[1].item() for x in r])), float(np.mean([x[2].item() for x in r])))
        # stage teacher: accuracy of ranking the real chunk first; stage head: teacher-energy gain of the pick (lower = better)
        score = np.mean([v[0] for v in res.values()]) if args.stage == "teacher" else -np.mean([v[1] for v in res.values()])
        print(f"VAL step={step} " + " ".join(f"{k}: acc={a:.3f} gain={g:+.4f}" for k, (a, g) in res.items()) + f" score={score:.4f}", flush=True)
        log.write(json.dumps({"step": step, **{f"{k}_acc": a for k, (a, g) in res.items()},
                              **{f"{k}_gain": g for k, (a, g) in res.items()}}) + "\n"); log.flush()
        if score > best:
            best = score
            fn = "energy_teacher.pt" if args.stage == "teacher" else "energy_head.pt"
            torch.save({"state": head.state_dict(), "d": d, "step": step, "C": DS["embody"].args.C}, f"{args.out}/{fn}")
print("done", flush=True)
