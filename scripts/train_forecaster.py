"""Train the speech forecaster (audio_forecaster.py) on streamed HuBERT features: at each frame, predict the next K
frames from the past only. Loss: MSE in feature-normalised space over the K horizons (frames past the window end are
masked). Validation reports, per horizon, the cosine similarity of the forecast to the true future frame next to the
"copy the last frame" baseline, so a forecast that only repeats the present is visible.

usage: python scripts/train_forecaster.py --out <dir> [--data outputs/stream-f24-nov04,outputs/stream-lower24-nov04]
       [--K 7] [--steps 6000] [--batch 32]
"""
import os, sys, json, time, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
import torch.nn.functional as F
from audio_forecaster import Forecaster

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--data", default="outputs/stream-f24-nov04,outputs/stream-lower24-nov04")
ap.add_argument("--K", type=int, default=7); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default="", help="start from another forecaster.pt (e.g. the Embody one for BEAT2)")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
TR = [np.load(f"{d}/train_hubert.npy", mmap_mode="r") for d in args.data.split(",")]
VA = np.load(f"{args.data.split(',')[0]}/val_hubert.npy", mmap_mode="r")
Fn = TR[0].shape[1]
rng = np.random.RandomState(0)
sample = np.asarray(TR[0][np.sort(rng.choice(len(TR[0]), 500, replace=False))]).reshape(-1, 1024).astype(np.float32)
mean, std = sample.mean(0), sample.std(0) + 1e-4
model = Forecaster(K=args.K, F=Fn, feat_mean=mean, feat_std=std).cuda()
if args.init:                                   # keeps the init's own feature normaliser (as speech_to_motion_small --init)
    model.load_state_dict(torch.load(args.init, map_location="cpu")["state"])
    fm0, fs0 = model.feat_mean.cpu().numpy(), model.feat_std.cpu().numpy(); mean, std = fm0, fs0
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))
fm, fs = torch.as_tensor(mean, device="cuda"), torch.as_tensor(std, device="cuda")


def targets(x):
    """(B, F, 1024) -> (B, F, K, 1024) true frames i+1..i+K and a (F, K) validity mask"""
    B, Fr, _ = x.shape
    idx = torch.arange(Fr, device=x.device)[:, None] + torch.arange(1, args.K + 1, device=x.device)[None]
    ok = idx < Fr
    return x[:, idx.clamp(max=Fr - 1)], ok


def loss_fn(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model(x)
    y, ok = targets(x)
    pn, yn = (p.float() - fm) / fs, (y.float() - fm) / fs
    return ((pn - yn) ** 2).mean(-1)[:, ok].mean(), p.float(), y.float(), ok


@torch.no_grad()
def validate():
    model.eval()
    cos_f, cos_c = np.zeros(args.K), np.zeros(args.K); n = 0; ls = []
    for i in range(0, min(128, len(VA)), 16):
        x = torch.as_tensor(np.asarray(VA[i:i + 16]), device="cuda").float()
        l, p, y, ok = loss_fn(x); ls.append(l.item())
        last = x[:, :, None].expand_as(y)
        for k in range(args.K):
            m = ok[:, k]
            cos_f[k] += F.cosine_similarity(p[:, m, k], y[:, m, k], dim=-1).mean().item()
            cos_c[k] += F.cosine_similarity(last[:, m, k], y[:, m, k], dim=-1).mean().item()
        n += 1
    model.train()
    return float(np.mean(ls)), cos_f / n, cos_c / n


t0, best = time.time(), 1e9
log = open(f"{args.out}/train_log.jsonl", "a")
for step in range(1, args.steps + 1):
    part = np.random.randint(len(TR))
    rows = np.sort(np.random.choice(len(TR[part]), args.batch, replace=False))
    x = torch.as_tensor(np.asarray(TR[part][rows]), device="cuda").float()
    loss = loss_fn(x)[0]
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 100 == 0:
        print(f"STEP {step} loss={loss.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    if step % args.val_every == 0 or step == args.steps:
        vl, cf, cc = validate()
        print(f"VAL step={step} mse={vl:.4f} cos_forecast={' '.join(f'{v:.3f}' for v in cf)} cos_copylast={' '.join(f'{v:.3f}' for v in cc)}", flush=True)
        log.write(json.dumps({"step": step, "val_mse": vl, "cos_f": cf.tolist(), "cos_copy": cc.tolist()}) + "\n"); log.flush()
        if vl < best:
            best = vl
            torch.save({"state": model.state_dict(), "cfg": model.cfg, "step": step}, f"{args.out}/forecaster.pt")
print(f"done, best val mse {best:.4f}", flush=True)
