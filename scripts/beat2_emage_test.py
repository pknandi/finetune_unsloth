"""EMAGE (CVPR 2024) on the BEAT2 speaker-2 test recordings, as PantoMatrix train_emage_audio.inference_fn
does it: whole recording at once, the real motion passed as masked_motion (EMAGE uses its first frames as
the seed). Foot contact from <beat2_dir>/footcontact (PantoMatrix datasets/foot_contact.py) when present, else zero. Runs in the EMAGE venv.
Writes <out>/<id>_output.npz (beat_format_save: poses (T,165), trans, expressions) for scripts/beat2_score.py.

usage: python scripts/beat2_emage_test.py <beat2_dir> <recordings.csv> <out_dir> [--gen <generator dir>]
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
import librosa

ap = argparse.ArgumentParser()
ap.add_argument("beat2"); ap.add_argument("recs"); ap.add_argument("out")
ap.add_argument("--code", default="/data2/home/mubtasim/ext/PantoMatrix")
ap.add_argument("--weights", default="/data2/home/mubtasim/ext/emage_work/hf/emage_audio")
ap.add_argument("--gen", default="", help="generator dir (default: the pretrained one in --weights)")
ap.add_argument("--split", default="test")
args = ap.parse_args()
sys.path.insert(0, args.code)
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
from emage_utils.motion_io import beat_format_save
import emage_utils.rotation_conversions as rc

dev = torch.device("cuda")
wd = args.weights
vq = EmageVQModel(
    face_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/face"),
    upper_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/upper"),
    lower_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/lower"),
    hands_model=EmageVQVAEConv.from_pretrained(f"{wd}/emage_vq/hands"),
    global_model=EmageVAEConv.from_pretrained(f"{wd}/emage_vq/global")).to(dev).eval()
model = EmageAudioModel.from_pretrained(args.gen or wd).to(dev).eval()
cfg = model.cfg
os.makedirs(args.out, exist_ok=True)
ids = [r["id"] for r in csv.DictReader(open(args.recs)) if r["split"] == args.split]
for rid in ids:
    audio, _ = librosa.load(f"{args.beat2}/wave16k/{rid}.wav", sr=cfg.audio_sr)
    audio = torch.from_numpy(audio).to(dev)[None]
    d = np.load(f"{args.beat2}/smplxflame_30/{rid}.npz", allow_pickle=True)
    poses = torch.from_numpy(d["poses"]).float().to(dev)[None]
    trans = torch.from_numpy(d["trans"]).float().to(dev)[None]
    bs, t, _ = poses.shape
    p6 = rc.axis_angle_to_rotation_6d(poses.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
    fc = f"{args.beat2}/footcontact/{rid}.npy"
    fc = torch.from_numpy(np.load(fc)).float().to(dev)[None] if os.path.exists(fc) else torch.zeros(bs, t, 4, device=dev)
    masked = torch.cat([p6, trans, fc], -1)
    with torch.no_grad():
        lat = model.inference(audio, torch.zeros(1, 1, dtype=torch.long, device=dev), vq, masked_motion=masked)
        get = lambda part, l, c: (lat[f"rec_{part}"] if l > 0 and c == 0 else None)
        idx = lambda part, c: (torch.max(F.log_softmax(lat[f"cls_{part}"], dim=2), dim=2)[1] if c > 0 else None)
        out = vq.decode(face_latent=get("face", cfg.lf, cfg.cf), upper_latent=get("upper", cfg.lu, cfg.cu),
                        lower_latent=get("lower", cfg.ll, cfg.cl), hands_latent=get("hands", cfg.lh, cfg.ch),
                        face_index=idx("face", cfg.cf), upper_index=idx("upper", cfg.cu),
                        lower_index=idx("lower", cfg.cl), hands_index=idx("hands", cfg.ch),
                        get_global_motion=True, ref_trans=trans[:, 0])
    n = out["motion_axis_angle"].shape[1]
    beat_format_save(f"{args.out}/{rid}_output.npz", out["motion_axis_angle"].cpu().numpy().reshape(n, -1),
                     upsample=30 // cfg.pose_fps, expressions=out["expression"].cpu().numpy().reshape(n, -1),
                     trans=out["trans"].cpu().numpy().reshape(n, -1))
    print(f"{rid}: {n} frames (gt {t})", flush=True)
