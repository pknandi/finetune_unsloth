"""Does the teacher's token score track good motion? (check Q1 of Nguyen et al. 2026, "Privileged likelihood is not
automatically value", before trusting on-policy distillation). For saved student rollouts (aligned_eval.py output on
the streamed eval set), each rollout gets
  score   = mean over frames after the 1 s seed of sum over RVQ levels of log p_teacher(token) - log p_student(token)
            (teacher: offline model on full-window audio; student: online model on streamed audio)
  outcome = speech match of that rollout from the eval matcher: margin (own-audio cosine minus mean cosine to the
            other clips' audio at the same time, 3 crops) and top-5 hit rate among the n clips
and we report the Spearman correlation of score with margin and with motion speed (mean |frame difference| of pose), and the AUC of score for "top-5 hit on >= 2 of 3 crops".
AUC ~0.5 = the teacher's score says nothing about which rollouts follow the speech.

usage: python scripts/teacher_score_check.py <student_ckpt> <teacher_ckpt> <eval_dir> [modes, comma list]
"""
import os, sys, re, csv, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts"); os.chdir(REPO)
import numpy as np
import torch
import torch.nn.functional as F
from speech_to_motion_small import load_trained
from audio_motion_match import Match, motion_feats, crops, STARTS, NA, NM, A_FPS, FPS_M

st_ck, te_ck, ev = sys.argv[1:4]
modes = (sys.argv[4] if len(sys.argv) > 4 else "sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2").split(",")
S_DATA, T_DATA, M_DATA = "outputs/stream-hubert-nov04", "outputs/run-full-dense-rel-oct28", "outputs/run-full-q8-oct21/aligned_data"
student, teacher = load_trained(st_ck)[0].cuda().eval(), load_trained(te_ck)[0].cuda().eval()
fs_all = np.load(f"{S_DATA}/val_hubert.npy", mmap_mode="r"); ft_all = np.load(f"{T_DATA}/val_hubert.npy", mmap_mode="r")
aud = np.load(f"{M_DATA}/val_hubert.npy", mmap_mode="r")
stt = torch.load("outputs/fix-oct21/match/match.pt", map_location="cpu")["state"]
mt = Match(stt["a_mean"], stt["a_std"], stt["m_mean"], stt["m_std"]).cuda(); mt.load_state_dict(stt); mt.eval()


@torch.no_grad()
def tok_score(i, ids):
    c = torch.as_tensor(ids, device="cuda")[None]
    ls = F.log_softmax(student(torch.as_tensor(np.asarray(fs_all[i]), device="cuda")[None], c).float(), -1)[0]
    lt = F.log_softmax(teacher(torch.as_tensor(np.asarray(ft_all[i]), device="cuda")[None], c).float(), -1)[0]
    g = lambda l: l.gather(-1, c[0][..., None])[..., 0]                         # (T, Q) log p of the sampled codes
    return (g(lt) - g(ls))[15:].sum(-1).mean().item()


rows = []
for mode in modes:
    files = {int(re.match(r"val(\d+)_", os.path.basename(p)).group(1)): p
             for p in glob.glob(f"{ev}/val*_{mode}_ids.npy")}
    idx = sorted(files)
    with torch.no_grad():
        ea = mt.embed_audio(torch.as_tensor(np.stack([crops(np.asarray(aud[i], np.float32), STARTS, NA, A_FPS) for i in idx]),
                                            device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
        em = mt.embed_motion(torch.as_tensor(np.stack([crops(motion_feats(np.load(files[i].replace("_ids", ""))[:300]), STARTS, NM, FPS_M)
                                                       for i in idx]), device="cuda").flatten(0, 1)).view(len(idx), len(STARTS), -1)
        cross = torch.einsum("nkd,mkd->knm", em, ea)                                  # (3, n motion, n audio)
        own = cross.diagonal(dim1=1, dim2=2)                                            # (3, n)
        margin = (own - (cross.sum(-1) - own) / (len(idx) - 1)).mean(0).cpu().numpy()
        hits = ((cross > own[..., None]).sum(-1) < 5).float().mean(0).cpu().numpy()
    for k, i in enumerate(idx):
        mo = np.load(files[i].replace("_ids", ""))[30:300, 3:156]
        rows.append((tok_score(i, np.load(files[i])), margin[k], hits[k], np.abs(np.diff(mo, axis=0)).mean()))
sc, mg, ht, sp = map(np.array, zip(*rows))
rk = lambda x: np.argsort(np.argsort(x))
spear = np.corrcoef(rk(sc), rk(mg))[0, 1]
pos, neg = sc[ht >= 2 / 3], sc[ht < 2 / 3]
auc = ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else float("nan")
print(f"TEACHERCHECK eval={ev} n={len(sc)} spearman(score,margin)={spear:+.3f} AUC(score->top5 hit)={auc:.3f} "
      f"pos={len(pos)} neg={len(neg)} mean_score={sc.mean():+.3f} spearman(score,motion_speed)={np.corrcoef(rk(sc), rk(sp))[0, 1]:+.3f}", flush=True)
