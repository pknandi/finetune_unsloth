"""A causal decoder for a trained RVQ motion tokenizer (contribution B: zero look-ahead output). The encoder, codebooks
and so every motion token stay as they are, so the generators need no retraining; only the decoder is replaced by
vqvae_motion_tokenizer.CausalDecoder, trained to reproduce the old symmetric decoder's output on the generators' own
token windows (<data>/train_motion.npy) with the tokenizer's loss (channel- and motion-energy-weighted L1 + velocity) plus
an acceleration term, since the online output can no longer be smoothed with future frames. (The tokenizer's
train_feats.npy is not usable: it predates the 6d_rel orientation the tokens were made with; the old decoder misses it
by L1 0.43.) Held out: <data>/val_motion.npy; prints the causal decoder's L1 to the old decoder's output each epoch.

usage: python scripts/train_causal_decoder.py <tok_dir> <out_tok_dir> --data <dense_dir>[,<dense_dir2>] [--epochs 8]
       [--hand_weight 0.3] [--acc_w 1.0] [--future 0] [--dec_hidden 0] [--jerk_w 0]
writes <out_tok_dir>/tokenizer.pt (+ normalizer.npz copied), loadable by VQVAETokenizer.load
"""
import os, sys, shutil, time, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import torch
from vqvae_motion_tokenizer import VQVAETokenizer, CausalDecoder, Normalizer, feature_channel_weights

ap = argparse.ArgumentParser()
ap.add_argument("tok_dir"); ap.add_argument("out_dir"); ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--hand_weight", type=float, default=0.3)
ap.add_argument("--acc_w", type=float, default=1.0)
ap.add_argument("--jerk_w", type=float, default=0.0, help="weight of a third-difference (jerk) term, against causal jitter")
ap.add_argument("--future", type=int, default=0, help="tokens of decoder look-ahead (0 = zero look-ahead)")
ap.add_argument("--dec_hidden", type=int, default=0, help="decoder width (default: the tokenizer's)")
args = ap.parse_args()
torch.manual_seed(0)
os.makedirs(args.out_dir, exist_ok=True)
shutil.copy(f"{args.tok_dir}/normalizer.npz", f"{args.out_dir}/normalizer.npz")
vq = VQVAETokenizer(); vq.load(f"{args.tok_dir}/tokenizer.pt"); m = vq.model.eval()
for p in m.parameters():
    p.requires_grad_(False)
old = m.decoder
input_dim, latent_dim, hidden_dim = old[-1].out_channels, m.encoder[4].out_channels, m.encoder[0].out_channels
dec = CausalDecoder(latent_dim, args.dec_hidden or hidden_dim, input_dim, args.future).cuda()
norm = Normalizer(); norm.load(f"{args.tok_dir}/normalizer.npz")
cw = torch.as_tensor(feature_channel_weights(norm.pose_repr, args.hand_weight), device="cuda").view(1, -1, 1)
tr_c = np.concatenate([np.load(f"{d}/train_motion.npy") for d in args.data.split(",")]).astype(np.int64)
va_c = np.load(f"{args.data.split(',')[0]}/val_motion.npy").astype(np.int64)
tr, va = np.arange(len(tr_c)), np.arange(len(va_c))
print(f"token windows train {len(tr_c)} val {len(va_c)}, slots {tr_c.shape[1]}, levels {tr_c.shape[2]}", flush=True)


def batch(idx, codes):
    c = torch.as_tensor(codes[np.sort(idx)], device="cuda")
    with torch.no_grad():
        q = m.quantizer.get_output_from_indices(c).permute(0, 2, 1)
        return old(q), q                                            # target: the old decoder's output


def loss_fn(y, x):
    with torch.no_grad():
        e = torch.zeros(x.shape[0], 1, x.shape[2], device="cuda")
        e[:, :, 1:] = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(1, keepdim=True); e[:, :, 0] = e[:, :, 1]
        fw = 1.0 + 5.0 * e / (e.mean(2, keepdim=True) + 1e-6)
    d1 = lambda z: z[:, :, 1:] - z[:, :, :-1]
    rec = (cw * fw * (y - x).abs()).mean()
    vel = (cw * fw[:, :, 1:] * (d1(y) - d1(x)).pow(2)).mean()
    acc = (cw * fw[:, :, 2:] * (d1(d1(y)) - d1(d1(x))).pow(2)).mean()
    out = rec + vel + args.acc_w * acc
    if args.jerk_w:
        jk = (cw * fw[:, :, 3:] * (d1(d1(d1(y))) - d1(d1(d1(x)))).pow(2)).mean()
        out = out + args.jerk_w * jk
    return out


@torch.no_grad()
def val():
    ln, jo, jn = [], [], []
    jerk = lambda z: (z[:, :, 3:] - 3 * z[:, :, 2:-1] + 3 * z[:, :, 1:-2] - z[:, :, :-3]).abs().mean().item()
    for c in range(0, len(va), args.batch):
        yo, q = batch(va[c:c + args.batch], va_c)
        yn = dec(q)
        ln.append((yn - yo).abs().mean().item()); jo.append(jerk(yo)); jn.append(jerk(yn))
    return np.mean(ln), np.mean(jo), np.mean(jn)


opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.0)
steps = args.epochs * (len(tr) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
best, t0, step = 1e9, time.time(), 0
rng = np.random.default_rng(0)
for ep in range(args.epochs):
    perm = rng.permutation(tr)
    for c in range(0, len(perm) - args.batch + 1, args.batch):
        x, q = batch(perm[c:c + args.batch], tr_c)
        loss = loss_fn(dec(q), x)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 200 == 0:
            print(f"STEP {step}/{steps} loss={loss.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
    ln, jo, jn = val()
    print(f"VAL epoch={ep + 1} L1 causal-to-old={ln:.4f} | jerk old={jo:.4f} causal={jn:.4f}", flush=True)
    if ln < best:
        best = ln
        st = {k: v for k, v in m.state_dict().items() if not k.startswith("decoder.")}
        st.update({f"decoder.{k}": v for k, v in dec.state_dict().items()})
        torch.save(st, f"{args.out_dir}/tokenizer.pt")
print(f"done, best held-out L1 causal-to-old {best:.4f}", flush=True)
