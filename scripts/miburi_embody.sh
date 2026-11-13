#!/bin/bash
# 2026-11-10: MIBURI zero-shot on Embody-3D, fresh eval groups 1-7 (Table 3). MIBURI released only BEAT2-trained
# checkpoints, so this is its all-speaker BEAT2 model (allspk_release_gtdm3_exp / last_1720) run through its own Embody-3D
# pipeline: our 420 windows (from the Super Star groups root) in MIBURI's Embody layout with Whisper word transcripts
# (scripts/miburi_embody_prep.py), build_hdf5_embody3d in 10 s chunk mode (= our eval window), its test.py with
# --dataset_ratio full_embody3d (MIBURI_FORCE_BEATX_EVAL: its Embody FID nets are not released; we score with our tools),
# outputs resampled 25 -> 30 fps and turned back to each window's heading (scripts/superstar_embody_bridge.py), then
# scored like every model, per group. Every stage skips work already done.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/miburi_embody.sh
cd /home/mubtasim/speech-motion || exit 1
M=/data2/home/mubtasim/ext/miburi; E=/data2/home/mubtasim/ext/miburi_eval; PY=/data2/home/mubtasim/envs/miburi/bin/python
EPY=/data2/home/mubtasim/venvs/emage/bin/python; R=/data2/home/mubtasim/ext/superstar_embody; F=outputs/fix-oct21; DG=$E/embody_g
[ -f $DG/.prep_done ] || { $PY scripts/miburi_embody_prep.py $DG $R/groups/all.txt 2>&1 | grep -E '^[0-9]+/|^done|rror' && touch $DG/.prep_done; }
[ -f $DG/.prep_done ] || exit 1
for d in $DG/g*; do ln -sfn $d $M/datasets/embody_3d_aiagent/$(basename $d); done
cd $M
T=experiments/allspk_release_gtdm3_exp/test_1720
# 2026-11-11: MIBURI hit a device-side assert after ~6 windows in one long run (state builds up; the failing window
# runs fine alone), so it runs in chunks of 5 windows, one process each; a chunk is skipped when all its outputs exist
ls -d $DG/g* | sort > $E/all_dirs.txt; split -l 5 -d -a 3 $E/all_dirs.txt $E/chunk_
for c in $E/chunk_*; do
    k=$(basename $c); CD=$E/chunks/$k; mkdir -p $CD
    miss=0; for d in $(cat $c); do b=$(basename $d); s=$(ls $d); [ -d $T/$b+${s}_C0 ] || miss=1; ln -sfn $d $CD/$b; done
    [ $miss = 0 ] && continue
    [ -f $CD.hdf5 ] || $PY -m scripts.trainers.dataloaders.embody3d_dyadic.build_hdf5_embody3d --data_dir $CD/ --hdf5_path $CD.hdf5 \
        --process_multimodal_signals --mimi_codebooks 8 --train_ratio 0 --val_ratio 0 > $CD.build.log 2>&1
    MIBURI_FORCE_BEATX_EVAL=1 $PY $E/run_test.py --config experiments/allspk_release_gtdm3_exp/config.yaml \
        --test_ckpt ./experiments/allspk_release_gtdm3_exp/last_1720.safetensors --embody3d_cache_path $CD.hdf5 \
        --dataset_ratio full_embody3d --batch_size 1 --save True --disable_filtering True > $CD.test.log 2>&1
done
echo "MIBURI outputs: $(ls -d $T/g*+*_C0 | wc -l)"
cd /home/mubtasim/speech-motion
$PY - $M/$T $R $E/dumps <<'EOF'
import sys, os, glob, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
T, R, out = sys.argv[1:]
for k in range(1, 8):
    os.makedirs(f"{out}/g{k}", exist_ok=True)
    for rid in open(f"{R}/groups/g{k}.txt").read().split():
        sid, subj, base = rid.split("_")
        f = f"{T}/{base}+{subj}_C0/pred.npz"
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True); P, tr = d["poses"].astype(np.float64), d["trans"].astype(np.float64)
        n = len(P); t_in = np.arange(n) / 25.0; t_out = np.clip(np.arange(300) / 30.0, 0, t_in[-1])
        Q = np.stack([Slerp(t_in, Rot.from_rotvec(P[:, j]))(t_out).as_rotvec() for j in range(P.shape[1])], 1)
        tr30 = np.stack([np.interp(t_out, t_in, tr[:, c]) for c in range(3)], 1)
        np.savez(f"{out}/g{k}/{rid}_output.npz", poses=Q.reshape(300, -1).astype(np.float32), trans=tr30.astype(np.float32))
    print(f"group {k}: {len(glob.glob(f'{out}/g{k}/*_output.npz'))} windows", flush=True)
EOF
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH; D=outputs/run-full-q8-oct21
for k in 1 2 3 4 5 6 7; do
    O=$F/e60_miburi_grp$k
    grep -q '^MATCH mode=miburi ' $O.log 2>/dev/null && continue
    rm -rf $O; $EPY scripts/superstar_embody_bridge.py $E/dumps/g$k $O miburi 2>&1 | grep -iE "rror|windows" | tail -1
    python scripts/motion_dist_metrics.py $O $D/aligned_data 2>&1 | grep -E "^DIST " | tee $O.log
    python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $O 2>&1 | grep MATCH | tee -a $O.log
done
echo 0 > $F/miburi_embody.exit
