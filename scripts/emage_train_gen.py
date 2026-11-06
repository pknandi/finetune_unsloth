"""Train EMAGE's audio-to-motion generator (PantoMatrix models.emage_audio.EmageAudioModel, CVPR 2024) on
Embody-3D (scripts/emage_export.py), with the part VQ-VAEs fine-tuned by scripts/emage_train_vq.py.
Runs in the EMAGE venv. Follows PantoMatrix train_emage_audio.py (64-frame crops, 4 seed frames, three
passes per step: seed only / random mask + audio / random mask without audio; Adam 1.5e-4 constant,
batch 56) with two fixes: gradients are clipped after backward (the original clips before it, a no-op),
and the random mask ratio ramps 0.05 -> 1 over 54k steps (the original's iteration/135*400 reaches 1
at step 1; 135*400 looks intended as the denominator). Face is not trained (Embody-3D has none).

usage: python scripts/emage_train_gen.py <export_dir> <vq_dir> <save_dir> [--steps 100000 --bs 56]
Saves <save_dir>/best (lowest val loss, from_pretrained dir) and <save_dir>/last.pt (resume).
"""
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("vq"); ap.add_argument("save")
ap.add_argument("--code", default="/data2/home/mubtasim/ext/PantoMatrix")
ap.add_argument("--config", default="/data2/home/mubtasim/ext/emage_work/hf/emage_audio/config.json")
ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--bs", type=int, default=56)
ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--val_every", type=int, default=2000)
args = ap.parse_args()
sys.path.insert(0, args.code)
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
from models.emage_audio.configuration_emage_audio import EmageAudioConfig
import emage_utils.rotation_conversions as rc

L, W, SPF = 64, 300, 16000 // 30
dev = torch.device("cuda")
torch.manual_seed(0); np.random.seed(0)


class Crops(torch.utils.data.Dataset):
    def __init__(self, split, fixed=False):
        d = args.data
        self.p = np.load(f"{d}/{split}_pose.npy", mmap_mode="r")
        self.t = np.load(f"{d}/{split}_trans.npy", mmap_mode="r")
        self.c = np.load(f"{d}/{split}_contact.npy", mmap_mode="r")
        self.a = np.load(f"{d}/{split}_audio.npy", mmap_mode="r")
        self.ok = np.where(np.load(f"{d}/{split}_ok.npy"))[0]
        self.fixed = fixed

    def __len__(self):
        return len(self.ok)

    def __getitem__(self, i):
        k = self.ok[i]
        s = (W - L) // 2 if self.fixed else np.random.randint(0, W - L + 1)
        return (torch.from_numpy(self.p[k, s:s + L].astype(np.float32)), torch.from_numpy(self.t[k, s:s + L].copy()),
                torch.from_numpy(self.c[k, s:s + L].astype(np.float32)),
                torch.from_numpy(self.a[k, s * SPF:(s + L) * SPF].astype(np.float32) / 32767))


vq = EmageVQModel(
    face_model=EmageVQVAEConv.from_pretrained(f"{args.vq}/emage_vq/face"),
    upper_model=EmageVQVAEConv.from_pretrained(f"{args.vq}/emage_vq/upper"),
    lower_model=EmageVQVAEConv.from_pretrained(f"{args.vq}/emage_vq/lower"),
    hands_model=EmageVQVAEConv.from_pretrained(f"{args.vq}/emage_vq/hands"),
    global_model=EmageVAEConv.from_pretrained(f"{args.vq}/emage_vq/global")).to(dev).eval()
for p in vq.parameters():
    p.requires_grad = False
cfg = EmageAudioConfig(**{k: v for k, v in json.load(open(args.config)).items() if not k.startswith("_")})
model = EmageAudioModel(cfg).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
nll = nn.NLLLoss()
c = cfg


def losses(out, lat, idx):
    """PantoMatrix get_rec_loss + get_cls_loss without the face terms"""
    l = 0.0
    for k in ("upper", "hands", "lower"):
        l = l + getattr(c, f"l{k[0]}") * F.mse_loss(out[f"rec_{k}"], lat[k])
        l = l + getattr(c, f"c{k[0]}") * nll(F.log_softmax(out[f"cls_{k}"], 2).permute(0, 2, 1), idx[k])
    return l


def step_loss(batch, it, train=True):
    pose, trans, contact, audio = [t.to(dev, non_blocking=True) for t in batch]
    B = len(pose)
    aa = torch.zeros(B, L, 55, 3, device=dev)
    aa[:, :, :22] = pose[:, :, :66].reshape(B, L, 22, 3); aa[:, :, 25:] = pose[:, :, 66:].reshape(B, L, 30, 3)
    r6 = rc.axis_angle_to_rotation_6d(aa).reshape(B, L, 330)
    expr = torch.zeros(B, L, 100, device=dev)
    with torch.no_grad():
        idx = vq.map2index(r6, expr, tar_contact=contact, tar_trans=trans)
        lat = vq.map2latent(r6, expr, tar_contact=contact, tar_trans=trans)
    mm = torch.cat([r6, trans, contact], 2)
    spk = torch.zeros(B, 1, dtype=torch.long, device=dev)
    mask = torch.ones_like(mm); mask[:, :c.seed_frames] = 0
    seed = losses(model(audio, spk, masked_motion=mm, mask=mask, use_audio=True), lat, idx)
    if not train:
        return seed
    ratio = min(1.0, it / 54000) * 0.95 + 0.05
    rmask = (torch.rand(B, L, mm.shape[2], device=dev) < ratio).float()
    ra = losses(model(audio, spk, masked_motion=mm, mask=rmask, use_audio=True), lat, idx)
    rm = losses(model(audio, spk, masked_motion=mm, mask=rmask, use_audio=False), lat, idx)
    return seed + ra + rm


tr = torch.utils.data.DataLoader(Crops("train"), batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True,
                                 persistent_workers=True, pin_memory=True)
va = torch.utils.data.DataLoader(Crops("val", fixed=True), batch_size=64, num_workers=4)
os.makedirs(args.save, exist_ok=True)
it, best = 0, float("inf")
if os.path.exists(f"{args.save}/last.pt"):
    ck = torch.load(f"{args.save}/last.pt", map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it, best = ck["it"], ck["best"]
    print(f"resumed at step {it}", flush=True)
run = 0.0
while it < args.steps:
    for batch in tr:
        model.train()
        loss = step_loss(batch, it)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.99)
        opt.step(); it += 1
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 500 == 0:
            print(f"step {it} train {run:.4f}", flush=True)
        if it % args.val_every == 0 or it == args.steps:
            model.eval()
            with torch.no_grad():
                v = float(np.mean([step_loss(b, it, train=False).item() for b in va]))
            if v < best:
                best = v
                model.save_pretrained(f"{args.save}/best")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "best": best}, f"{args.save}/last.pt")
            print(f"VAL step {it} seed_loss {v:.4f} best {best:.4f}", flush=True)
        if it >= args.steps:
            break
print("done", flush=True)
