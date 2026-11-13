"""MIBURI zero-shot on Embody-3D, step 1 (2026-11-10): MIBURI's Embody-3D layout for our fresh eval windows. Reuses the
windows written for Super Star (scripts/superstar_embody_data.py --groups: 12.5 s, BEAT2-style npz at 30 fps, turned to
face +Z, 16 kHz wav), and writes <out>/<base>/<subject>/{smplx_25/<base>.npz, audio_separated/<base>.wav,
transcription/<base>.json} with base = g<val index>: motion resampled to 25 fps (slerp per joint, linear translation),
and Whisper word timestamps in MIBURI's transcript format ({"segments": [{"words": [{"word", "start", "end"}]}]}).
Runs in MIBURI's env (transformers Whisper).

usage: python scripts/miburi_embody_prep.py <out_dir> [ids_file=/data2/home/mubtasim/ext/superstar_embody/groups/all.txt]
"""
import os, sys, json, shutil
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

out = sys.argv[1]
ids = open(sys.argv[2] if len(sys.argv) > 2 else "/data2/home/mubtasim/ext/superstar_embody/groups/all.txt").read().split()
ROOT = "/data2/home/mubtasim/ext/superstar_embody"
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0, return_timestamps="word")
import soundfile as sf


def to25(x, n_out):
    t_in, t_out = np.arange(len(x)) / 30.0, np.arange(n_out) / 25.0
    t_out = np.clip(t_out, 0, t_in[-1])
    J = x.reshape(len(x), -1, 3)
    o = np.stack([Slerp(t_in, R.from_rotvec(J[:, j]))(t_out).as_rotvec() for j in range(J.shape[1])], 1)
    return o.reshape(n_out, -1).astype(np.float32)


for k, rid in enumerate(ids):
    sid, subj, g = rid.split("_")
    base = g; d = f"{out}/{base}/{subj}"
    if os.path.exists(f"{d}/transcription/{base}.json"):
        continue
    for s in ("smplx_25", "audio_separated", "transcription"):
        os.makedirs(f"{d}/{s}", exist_ok=True)
    m = np.load(f"{ROOT}/smplxflame_30/{rid}.npz")
    n30 = len(m["poses"]); n25 = int(n30 * 25 / 30)
    tr = np.stack([np.interp(np.arange(n25) / 25.0, np.arange(n30) / 30.0, m["trans"][:, c]) for c in range(3)], 1).astype(np.float32)
    np.savez(f"{d}/smplx_25/{base}.npz", poses=to25(m["poses"], n25), trans=tr, betas=m["betas"],
             expressions=np.zeros((n25, 100), np.float32), mocap_frame_rate=25)
    shutil.copy(f"{ROOT}/wave16k/{rid}.wav", f"{d}/audio_separated/{base}.wav")
    wav, sr = sf.read(f"{d}/audio_separated/{base}.wav", dtype="float32")
    r = asr({"raw": wav, "sampling_rate": sr})
    words = [{"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]} for c in r.get("chunks", [])]
    words = [w for w in words if w["start"] is not None]
    for j, w in enumerate(words):                   # Whisper can leave the last end open: next start or +0.3 s
        if w["end"] is None:
            w["end"] = words[j + 1]["start"] if j + 1 < len(words) else min(w["start"] + 0.3, len(wav) / sr)
    json.dump({"segments": [{"words": words}]}, open(f"{d}/transcription/{base}.json", "w"))
    if k % 50 == 0:
        print(f"{k}/{len(ids)} {rid}: {len(words)} words", flush=True)
print("done", len(ids), flush=True)
