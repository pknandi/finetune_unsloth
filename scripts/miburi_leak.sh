#!/bin/bash
# 2026-11-08: measured audio look-ahead of MIBURI (Table 1), with its own released pipeline and checkpoint (as the BEAT2
# eval of 11-07: allspk_release_gtdm3_exp / last_1720, build_hdf5_beatx --full_sequence --process_multimodal_signals,
# scripts/test.py via miburi_eval/run_test.py). For 3 speaker-2 test recordings the audio after TAU s is replaced by
# noise (transcripts and motion unchanged, so this measures the audio channel only); MIBURI runs on the original twice
# (to see how deterministic it is) and on the perturbed audio once. Look-ahead = TAU - time of the first output frame
# that differs from the original (beyond the original-vs-original difference). Writes outputs/fix-oct21/miburi_leak.log.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/miburi_leak.sh
M=/data2/home/mubtasim/ext/miburi; E=/data2/home/mubtasim/ext/miburi_eval; L=$E/leak; PY=/data2/home/mubtasim/envs/miburi/bin/python
TAU=${TAU:-30}; IDS="2_scott_0_103_103 2_scott_0_111_111 2_scott_0_112_112"
OUT=/home/mubtasim/speech-motion/outputs/fix-oct21/miburi_leak.log
mkdir -p $L/orig/beat_english_v2.0.0/wave16k $L/pert/beat_english_v2.0.0/wave16k $L/runs
for v in orig pert; do for d in smplxflame_25 whisper_transcription weights; do
    ln -sfn $M/datasets/beat_english_v2.0.0/$d $L/$v/beat_english_v2.0.0/$d; done; done
{ echo "id,type"; for i in $IDS; do echo "$i,test"; done; } > $L/list.csv
$PY - $M/datasets/beat_english_v2.0.0/wave16k $L $TAU $IDS <<'EOF'
import sys, numpy as np, soundfile as sf
src, L, tau, ids = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
rng = np.random.default_rng(0)
for i in ids:
    w, sr = sf.read(f"{src}/{i}.wav", dtype="float32")
    sf.write(f"{L}/orig/beat_english_v2.0.0/wave16k/{i}.wav", w, sr)
    p = w.copy(); k = int(tau * sr); p[k:] = rng.normal(0, w.std(), p[k:].shape).astype(np.float32)
    sf.write(f"{L}/pert/beat_english_v2.0.0/wave16k/{i}.wav", p, sr)
print("wrote", len(ids), "clips, cut at", tau, "s")
EOF
cd $M || exit 1
for v in orig pert; do
    [ -f $L/$v.hdf5 ] || $PY -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx --data_dir $L/$v/beat_english_v2.0.0/ \
        --file_list_path $L/list.csv --hdf5_path $L/$v.hdf5 --full_sequence --process_multimodal_signals --mimi_codebooks 8 2>&1 | tail -2
done
T=experiments/allspk_release_gtdm3_exp/test_1720
[ -d $E/test_1720_backup ] || cp -r $T $E/test_1720_backup                 # the 11-07 BEAT2 outputs, kept
for run in origA:orig origB:orig pert:pert; do
    r=${run%%:*}; v=${run#*:}
    [ -d $L/runs/$r ] && continue
    for i in $IDS; do rm -rf $T/$i; done                                       # no stale outputs if a run fails
    $PY $E/run_test.py --config experiments/allspk_release_gtdm3_exp/config.yaml --test_ckpt ./experiments/allspk_release_gtdm3_exp/last_1720.safetensors \
        --beatx_cache_path $L/$v.hdf5 --dataset_ratio scott_beatx_fulllength --batch_size 1 --save True 2>&1 | grep -iE "error" | grep -v sqrtm | tail -2
    mkdir -p $L/runs/$r; for i in $IDS; do cp $T/$i/pred.npz $L/runs/$r/$i.npz; done
done
$PY - $L/runs $TAU $IDS <<'EOF' | tee $OUT
import sys, numpy as np
R, tau, ids = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
fps = 25
res = []
for i in ids:
    a, b, p = (np.load(f"{R}/{r}/{i}.npz") for r in ("origA", "origB", "pert"))
    k = "poses" if "poses" in a.files else a.files[0]
    A, B, P = a[k].reshape(len(a[k]), -1), b[k].reshape(len(b[k]), -1), p[k].reshape(len(p[k]), -1)
    n = min(len(A), len(B), len(P))
    noise = np.abs(A[:n] - B[:n]).max(1)                     # run-to-run difference on identical input
    d = np.abs(A[:n] - P[:n]).max(1)
    thr = max(1e-5, 10 * noise.max()) if noise.max() > 0 else 1e-5
    hit = np.where(d > thr)[0]
    t_first = hit[0] / fps if len(hit) else None
    print(f"MIBURI_LEAK {i} key={k} frames={n} orig_vs_orig_max={noise.max():.2e} first_change_s={t_first} "
          f"lookahead_s={(tau - t_first) if t_first is not None else 'none'}", flush=True)
    if t_first is not None: res.append(tau - t_first)
print(f"MIBURI_LEAK max_lookahead_s={max(res) if res else 'none'} tau={tau}", flush=True)
EOF
