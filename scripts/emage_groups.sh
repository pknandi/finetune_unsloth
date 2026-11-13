#!/bin/bash
# 2026-11-06: EMAGE (retrained on Embody-3D, outputs/emage-gen-oct28/best) on the fresh Embody eval groups 1-4, as in
# the 10-30 group-0 evaluation (own 4-frame seed), scored like our models (FD, speech match). --weights must be the
# Embody-finetuned part VQs (outputs/emage-vq-oct28) the generator was trained with; the default HF BEAT2 VQs give garbage.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/emage_groups.sh [groups]
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; D=outputs/run-full-q8-oct21/aligned_data; PY=/data2/home/mubtasim/venvs/emage/bin/python
for g in ${1:-1 2 3 4}; do
    E=$F/e60_emage_grp$g
    EVAL_GROUP=$g $PY scripts/emage_bridge.py outputs/run-full-dense-rel-oct28 $E 60 emage --weights outputs/emage-vq-oct28 --gen outputs/emage-gen-oct28/best 2>&1 | grep -E "rror" | tee $F/e60_emage_grp$g.log
    /data/home/mubtasim/conda-envs/motion/bin/python scripts/motion_dist_metrics.py $E $D 2>&1 | grep -E "^DIST " | tee -a $F/e60_emage_grp$g.log
    /data/home/mubtasim/conda-envs/motion/bin/python scripts/audio_motion_match.py score $F/match/match.pt $D $E 2>&1 | grep -E "MATCH" | tee -a $F/e60_emage_grp$g.log
done
