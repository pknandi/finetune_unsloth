"""Model soup (2026-11-08): average the weights of several distillation runs that start from the same student checkpoint
(same recipe, different training seeds / runs). On BEAT2 the runs' test FGD spreads 0.545-0.669 (one 15-clip test
set); a uniform weight average is one model of the same size and speed, often better and less run-dependent than any
single run. All inputs must share cfg. Writes <out>/small.pt.

usage: python scripts/soup.py <out_dir> <ckpt_dir> <ckpt_dir> ...
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch

out, ins = sys.argv[1], sys.argv[2:]
sts = [torch.load(f"{d}/small.pt", map_location="cpu") for d in ins]
norm = lambda c: {"onset": 0, "partner": 0, "any_shifts": None, "speakers": 0, "spk_default": 0, **c}          # keys added later default to off
assert all(norm(s["cfg"]) == norm(sts[0]["cfg"]) for s in sts), [s["cfg"] for s in sts]
avg = {}
for k, v in sts[0]["state"].items():
    avg[k] = (sum(s["state"][k].float() for s in sts) / len(sts)).to(v.dtype) if v.is_floating_point() else v
os.makedirs(out, exist_ok=True)
torch.save({"state": avg, "cfg": sts[0]["cfg"], "soup": ins}, f"{out}/small.pt")
print(f"soup of {len(ins)} -> {out}/small.pt", flush=True)
