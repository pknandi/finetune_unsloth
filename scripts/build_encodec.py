"""Audio-feature ablation: the same dense windows with EnCodec (24 kHz, continuous encoder output, the
features under the tokens of the original EnCodec -> Llama pipeline) instead of HuBERT-large layer 12.
128 dims at 75 fps, averaged to 25 fps (3 frames) and zero-padded to 1024 dims, written under the HuBERT
file names (<out>/{train,val}_hubert.npy) so speech_to_motion_small.py and scripts/aligned_eval.py run
unchanged (padded channels normalize to 0: std + 1e-4). The other files are symlinked from <src>.

usage: python scripts/build_encodec.py <src_dense_dir> <out_dir>
"""
import os, sys, csv, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speech_to_motion_aligned import read_window

src, out = map(os.path.abspath, sys.argv[1:3])
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith("_hubert.npy") and not os.path.exists(f"{out}/{f}"):
        os.symlink(f"{src}/{f}", f"{out}/{f}")
meta = json.load(open(f"{src}/meta.json"))
DUR = (150 + meta["lookahead"]) / 15
from transformers import EncodecModel
enc = EncodecModel.from_pretrained("facebook/encodec_24khz").cuda().eval().encoder

for split in ["val", "train"]:
    ws = [(r["audio_filename"], float(r["start_sec"])) for r in csv.DictReader(open(f"{src}/{split}_windows.csv"))]
    NF = np.load(f"{src}/{split}_hubert.npy", mmap_mode="r").shape[1]
    o = np.lib.format.open_memmap(f"{out}/{split}_hubert.npy", "w+", np.float16, (len(ws), NF, 1024))
    rd = lambda w: read_window(w[0], w[1], DUR, sr_out=24000)
    B = 32
    with ThreadPoolExecutor(16) as ex:
        nxt = ex.map(rd, ws[0:B])
        for i in range(0, len(ws), B):
            wavs = list(nxt)
            if i + B < len(ws):
                nxt = ex.map(rd, ws[i + B:i + 2 * B])
            with torch.no_grad():
                e = enc(torch.from_numpy(np.stack(wavs)).cuda()[:, None]).transpose(1, 2)     # (B, 790, 128)
            n = e.shape[1] // 3
            e = e[:, :3 * n].reshape(len(wavs), n, 3, 128).mean(2)
            e = torch.cat([e, e[:, -1:].expand(-1, max(0, NF - n), -1)], 1)[:, :NF]
            o[i:i + len(wavs), :, :128] = e.float().cpu().numpy().astype(np.float16)
            if i % (B * 200) == 0:
                print(f"{split} encodec {i}/{len(ws)}", flush=True)
    o.flush()
    print(f"{split}: {len(ws)} windows", flush=True)
