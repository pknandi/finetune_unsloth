"""Fine-tune EMAGE's part VQ-VAEs (upper body, hands, lower body + translation + foot contact) and its
global-motion autoencoder on Embody-3D (scripts/emage_export.py), starting from the BEAT2 weights. The
BEAT2 ones cannot represent sitting / kneeling / leaning poses (219 mm reconstruction on our 60 clips).
Face VQ is copied unchanged (Embody-3D has no face). Runs in the EMAGE venv.

Loss per part: MSE on the features (6D rotations normalized first) + on their velocity and acceleration + the VQ commitment loss (codes
unused for 500 steps are reset to random encoder outputs). Global AE target: lower-body 6D, then x
velocity (m/s), absolute y, z velocity, then foot contact, as models.emage_audio get_global_motion reads it;
only the x-velocity / y / z-velocity channels (the ones get_global_motion uses) are trained.

usage: python scripts/emage_train_vq.py <export_dir> <save_dir> [--steps 20000 --bs 256 --lr 1e-4]
Saves <save_dir>/emage_vq/{upper,hands,lower,global,face} (from_pretrained dirs); prints val rotation error (degrees) per part.
"""
import os, sys, argparse, shutil
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("save")
ap.add_argument("--code", default="/data2/home/mubtasim/ext/PantoMatrix")
ap.add_argument("--init", default="/data2/home/mubtasim/ext/emage_work/hf/emage_audio")
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-4)
args = ap.parse_args()
sys.path.insert(0, args.code)
from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageVQModel
import emage_utils.rotation_conversions as rc

L, W = 64, 300
dev = torch.device("cuda")
torch.manual_seed(0); np.random.seed(0)


class Crops(torch.utils.data.Dataset):
    def __init__(self, split):
        d = args.data
        self.p = np.load(f"{d}/{split}_pose.npy", mmap_mode="r")
        self.t = np.load(f"{d}/{split}_trans.npy", mmap_mode="r")
        self.c = np.load(f"{d}/{split}_contact.npy", mmap_mode="r")
        self.ok = np.where(np.load(f"{d}/{split}_ok.npy"))[0]

    def __len__(self):
        return len(self.ok)

    def __getitem__(self, i):
        k, s = self.ok[i], np.random.randint(0, W - L + 1)
        return (torch.from_numpy(self.p[k, s:s + L].astype(np.float32)), torch.from_numpy(self.t[k, s:s + L].copy()),
                torch.from_numpy(self.c[k, s:s + L].astype(np.float32)))


def parts(pose, trans, contact):
    """(B, L, 156) axis-angle -> dict of part features as EmageVQModel.spilt_inputs builds them"""
    B = len(pose)
    aa = torch.zeros(B, L, 55, 3, device=dev)
    aa[:, :, :22] = pose[:, :, :66].reshape(B, L, 22, 3); aa[:, :, 25:] = pose[:, :, 66:].reshape(B, L, 30, 3)
    r6 = rc.axis_angle_to_rotation_6d(aa).reshape(B, L, 330)
    x = vqm.spilt_inputs(r6, torch.zeros(B, L, 100, device=dev), tar_contact=contact, tar_trans=trans)
    v = torch.zeros(B, L, 3, device=dev)
    v[:, :-1] = ((trans[:, 1:] - trans[:, :-1]) * 30).clamp(-3, 3)      # tracking glitches reach 50 m/s
    x["global"] = torch.cat([x["lower"][..., :54], v[..., :1], trans[..., 1:2], v[..., 2:3], contact], 2)
    return x


NROT = {"upper": 13, "hands": 30, "lower": 9, "trans": 0}


def norm6(x, nrot):
    """the first nrot 6D rotations Gram-Schmidt normalized (decoders output unnormalized 6D)"""
    B, T, _ = x.shape
    r = rc.matrix_to_rotation_6d(rc.rotation_6d_to_matrix(x[..., :nrot * 6].reshape(B, T, nrot, 6))).reshape(B, T, -1)
    return torch.cat([r, x[..., nrot * 6:]], -1)


def angle_deg(rec, x, nrot):
    B, T, _ = x.shape
    a = rc.rotation_6d_to_matrix(rec[..., :nrot * 6].reshape(B, T, nrot, 6))
    b = rc.rotation_6d_to_matrix(x[..., :nrot * 6].reshape(B, T, nrot, 6))
    c = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(c.clamp(-1, 1))).mean().item()


def motion_loss(rec, x, k):
    rec = norm6(rec, NROT[k])
    l = F.mse_loss(rec, x)
    dr, dx = rec[:, 1:] - rec[:, :-1], x[:, 1:] - x[:, :-1]
    return l + F.mse_loss(dr, dx) + F.mse_loss(dr[:, 1:] - dr[:, :-1], dx[:, 1:] - dx[:, :-1])


w = args.init
nets = {k: EmageVQVAEConv.from_pretrained(f"{w}/emage_vq/{k}").to(dev) for k in ("upper", "hands", "lower")}
nets["global"] = EmageVAEConv.from_pretrained(f"{w}/emage_vq/global").to(dev)
vqm = EmageVQModel(face_model=None, upper_model=nets["upper"], hands_model=nets["hands"],
                   lower_model=nets["lower"], global_model=nets["global"])
opt = torch.optim.AdamW([p for n in nets.values() for p in n.parameters()], lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * s / args.steps)))
tr = torch.utils.data.DataLoader(Crops("train"), batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True,
                                 persistent_workers=True)
va = Crops("val")
last_used = {k: torch.zeros(256, device=dev) for k in ("upper", "hands", "lower")}


def step_vq(k, x, step):
    net = nets[k]
    z = net.encoder(x)
    emb, zq, idx, _ = net.quantizer(z)
    if net.training:
        last_used[k][idx.unique()] = step
        dead = (step - last_used[k] > 500).nonzero().flatten()
        if step % 100 == 0 and len(dead):
            with torch.no_grad():
                zf = z.reshape(-1, z.shape[-1])
                net.quantizer.embedding.weight.data[dead] = zf[torch.randint(len(zf), (len(dead),), device=dev)]
            last_used[k][dead] = step
    return motion_loss(net.decoder(zq), x, k) + emb, idx


@torch.no_grad()
def evaluate():
    for n in nets.values():
        n.eval()
    rng = np.random.default_rng(0)
    ks = np.arange(len(va))
    out = {k: [] for k in ("upper", "hands", "lower", "trans_contact_mse", "global_v_mse")}
    for b in range(0, len(ks), 128):
        items = [va[i] for i in ks[b:b + 128]]
        x = parts(*[torch.stack(t).to(dev) for t in zip(*items)])
        for k in ("upper", "hands", "lower"):
            idx = nets[k].map2index(x[k])
            out[k].append(angle_deg(nets[k].decode(idx), x[k], NROT[k]))
            if k == "lower":
                out["trans_contact_mse"].append(F.mse_loss(nets[k].decode(idx)[..., 54:], x[k][..., 54:]).item())
        g = nets["global"](x["lower"])["rec_pose"]
        out["global_v_mse"].append(F.mse_loss(g[..., 54:57], x["global"][..., 54:57]).item())
    for n in nets.values():
        n.train()
    return {k: float(np.mean(v)) for k, v in out.items()}


print("val (BEAT2 weights)", {k: round(v, 5) for k, v in evaluate().items()}, flush=True)
step = 0
while step < args.steps:
    for pose, trans, contact in tr:
        x = parts(pose.to(dev), trans.to(dev), contact.to(dev))
        loss, logs = 0.0, {}
        for k in ("upper", "hands", "lower"):
            l, idx = step_vq(k, x[k], step)
            loss = loss + l; logs[k] = l.item()
        lg = motion_loss(nets["global"](x["lower"])["rec_pose"][..., 54:57], x["global"][..., 54:57], "trans")
        loss = loss + lg; logs["global"] = lg.item()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for n in nets.values() for p in n.parameters()], 1.0)
        opt.step(); sched.step(); step += 1
        if step % 500 == 0:
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()), flush=True)
        if step % 5000 == 0 or step == args.steps:
            print(f"val step {step}", {k: round(v, 5) for k, v in evaluate().items()}, flush=True)
            for k, n in nets.items():
                n.save_pretrained(f"{args.save}/emage_vq/{k}")
        if step >= args.steps:
            break
if not os.path.exists(f"{args.save}/emage_vq/face"):
    shutil.copytree(f"{w}/emage_vq/face", f"{args.save}/emage_vq/face")
print("done", flush=True)
