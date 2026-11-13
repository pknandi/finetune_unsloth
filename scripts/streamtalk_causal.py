"""StreamTalk (Zhang et al., ECCV 2026; released code / speaker-2 checkpoint) under our online constraint (2026-11-11).
Released inference (Scripts/FM/TestFixedExpressions_keyposes.py) cuts the whole recording into 2 s windows (8-frame
overlap), runs bidirectional WavLM on each, generates each window's 60 frames at once, and averages each window's last 8
frames with the next window's first 8: every frame hears the rest of its window (up to ~1.73 s) plus the next window.
Here every output frame k of window n is generated from that window's WavLM features computed on the audio CUT at the
end of frame k plus LOOK s (the rest of the window replaced by silence): the window is sampled once per frame (their
initial sample, key-pose retrieval and key-pose refinement unchanged) and only frame k is kept; the next window is
seeded with the emitted frames, with no blending. LOOK = 0 is zero audio look-ahead (as UpBeat), 0.133 matches
UpBeat's 133 ms decoder. Runs in StreamTalk's env from its Scripts/FM directory (imports its modules unchanged).

usage: (cd <StreamTalk>/Scripts/FM && python /home/mubtasim/speech-motion/scripts/streamtalk_causal.py <wav> <ckpt>
        --example_motion_npz <gt npz> --pid 2_scott_0 --look 0 --output_file <out.npz> --wavlm_model <id> --smplx_model <npz>
        --retrieval_data_folder <smplxflame_30> --retrieval_csv_file <train_test_split.csv> --cfg_scale 3)
"""
import sys, os, argparse
sys.path.insert(0, os.getcwd())
import numpy as np, torch, librosa
import torch.nn.functional as F
import TestFixedExpressions_keyposes as ST
from Common.Util import NormalizeRot6d, rotation_6d_to_matrix, rotation_matrix_to_axis_angle, axis_angle_to_rotation_6d, matrix_to_rotation_6d
from Common.SMPLX_Util import global_to_local_batch, LocalToGlobalAA
from Common.SMPLXConfig import GetBodyPartIndices
from Net import SimpleSpeechModel
from Net import DiffusionDITNetPartsFixedExpressions2PostNormInteraction2 as MotionModel
from FindNearest import MakeKDTree

ap = argparse.ArgumentParser()
ap.add_argument("audio"); ap.add_argument("ckpt")
ap.add_argument("--example_motion_npz", required=True); ap.add_argument("--pid", default="2_scott_0")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cfg_scale", type=float, default=3.0)
ap.add_argument("--look", type=float, default=0.0, help="audio after the end of each frame + LOOK s is replaced by silence")
ap.add_argument("--output_file", required=True); ap.add_argument("--wavlm_model", required=True)
ap.add_argument("--smplx_model", required=True); ap.add_argument("--retrieval_data_folder", required=True)
ap.add_argument("--retrieval_csv_file", required=True)
args = ap.parse_args()
ST.setup_seed(args.seed)
hand, upper, lower, face = (GetBodyPartIndices(p, "rot6d") for p in ("hand", "upper", "lower", "face"))
trans_idx = np.array([330, 331, 332]); n_seed, L, B, SR = 8, 60, 1, 16000
speech = SimpleSpeechModel().cuda().requires_grad_(False).eval()
speech.load_state_dict(torch.load(ST.BUNDLED_FACE_CHECKPOINT, map_location="cpu"))
net = MotionModel().cuda().requires_grad_(False).eval(); net.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
smplx = ST.EvaluateSMPLX(args.smplx_model)
kd = MakeKDTree(pid=args.pid, data_folder=args.retrieval_data_folder, csv_file=args.retrieval_csv_file, smplx_model_path=args.smplx_model)
kd["poses"] = axis_angle_to_rotation_6d(torch.from_numpy(LocalToGlobalAA(kd["poses"])).float()).cuda()
kd["joints"] = torch.from_numpy(kd["joints"]).float().cuda().half()
pid = torch.LongTensor([int(args.pid.split("_")[0])]).cuda()
ex = np.load(args.example_motion_npz, allow_pickle=True)
betas = ex["betas"].astype(np.float32)
pose_prompt = torch.from_numpy(axis_angle_to_rotation_6d(LocalToGlobalAA(ex["poses"].astype(np.float32)[:L]))).cuda()[None].float()
trans_prompt = torch.from_numpy(ex["trans"].astype(np.float32)[:L]).cuda()[None].float()
fe = ST.AudioFeatureExtractor(args.wavlm_model).requires_grad_(False).cuda().eval()
audio = librosa.load(args.audio, sr=SR)[0]
win, stride = SR * 2, SR * 2 - int(SR / 30 * n_seed)
n_win = (len(audio) - win) // stride + 1


def feats(clip):
    iv = fe.processor(clip, return_tensors="pt", sampling_rate=SR).input_values.cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        h = fe.model.forward(iv).last_hidden_state
        return F.interpolate(h.permute(0, 2, 1), L, mode="linear", align_corners=True).permute(0, 2, 1)


def sample_window(hub, pose_prompt, trans_prompt):
    """their per-window procedure (initial sample, key-pose retrieval, key-pose refinement), unchanged"""
    pm = torch.full((B, L), True).cuda(); pm[:, :n_seed] = False
    tm = torch.full((B, L), True).cuda(); tm[:, :n_seed] = False
    ex_c = speech(hub, pid); ex_u = speech(hub, torch.zeros_like(pid)) if args.cfg_scale != 1.0 else None
    kw = dict(pid=pid, poses_prompt=pose_prompt.clone(), prompt_mask=pm, trans_prompt=trans_prompt, trans_mask=tm,
              hand_indices=hand, upper_indices=upper, lower_indices=lower, face_joint_indices=face, trans_indices=trans_idx,
              hubert=hub, expressions=ex_c)
    if ex_u is not None:
        kw["expressions_uncond"] = ex_u
    noise = torch.randn((B, L, 55 * 6 + 3)).cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        xl = net.sample(noise, normalize_func=NormalizeRot6d, cfg_scale=args.cfg_scale, **kw); xt = xl[-1]
    xt[..., :330] = NormalizeRot6d(xt[..., :330])
    lp = rotation_matrix_to_axis_angle(global_to_local_batch(rotation_6d_to_matrix(xt[..., :330].reshape(-1, 55, 6)))).reshape(B, -1, 165)
    j = smplx(betas=torch.from_numpy(betas).float().cuda(), poses=lp.reshape(-1, 165).cuda()).reshape((L, 1, -1))
    d, ids = torch.min(torch.abs(kd["joints"] - j[-8:].half()).sum(-1), dim=-1)   # = theirs on all 60, then [-8:] (less memory)
    k = int(torch.argmin(d)); np_ = rotation_6d_to_matrix(kd["poses"][ids[k]].reshape(1, 55, 6))
    go = rotation_6d_to_matrix(xt[0, 52 + k, :6].reshape(1, 6)); np_ = matrix_to_rotation_6d(torch.einsum("bij,bnjk->bnik", go, np_)).reshape(1, 330)
    pm2 = pm.clone(); pm2[:, 52 + k] = False; kw["poses_prompt"][:, 52 + k] = np_; kw["prompt_mask"] = pm2; xt[:, 52 + k, :330] = np_
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        xt = net.sample_with_kp_bk(noise, xl[5], 5, normalize_func=NormalizeRot6d, cfg_scale=args.cfg_scale, **kw)
    xt[..., :330] = NormalizeRot6d(xt[..., :330])
    return torch.cat([xt, ex_c], -1)


out = []
for n in range(n_win):
    s0 = n * stride; clip = audio[s0:s0 + win].copy(); frames = []
    for k in range(n_seed if n else 0, L):
        cut = int((k + 1) / 30 * SR + args.look * SR)          # audio heard by the end of frame k (+ LOOK)
        c = clip.copy(); c[cut:] = 0.0
        frames.append(sample_window(feats(c), pose_prompt, trans_prompt)[:, k])
    w = torch.stack(frames, 1)                                     # this window's emitted frames
    out.append(w)
    pose_prompt[:, :n_seed] = w[:, -n_seed:, :330]; trans_prompt[:, :n_seed] = w[:, -n_seed:, 330:333]
    print(f"window {n + 1}/{n_win}", flush=True)
rot = torch.cat(out, 1)
pp = rotation_matrix_to_axis_angle(global_to_local_batch(rotation_6d_to_matrix(rot[..., :330].reshape(-1, 55, 6)))).reshape(-1, 165)
np.savez(args.output_file, betas=betas, poses=pp.cpu().numpy(), expressions=rot[0, :, -100:].cpu().numpy(),
         trans=rot[0, :, 330:333].cpu().numpy(), model="smplx2020", gender="neutral", mocap_frame_rate=30)
print("saved", args.output_file, pp.shape, flush=True)
