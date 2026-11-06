#!/bin/bash
# 2026-11-05: 3 more draws (s3-s5) of the offline teacher F2 on the 60 eval windows, same pipeline as
# scripts/ol_stream_eval.sh but on full-window audio features. usage: CUDA_VISIBLE_DEVICES=1 scripts/eval60_f2_more.sh [name] [modes]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; D=outputs/run-full-q8-oct21; N=${1:-dt_F2_more}; E=$F/e60_$N
NO_VIDEO=1 python scripts/aligned_eval.py outputs/dt-F2-cfg-oct30/best outputs/run-full-dense-rel-oct28 outputs/tok-dense-rel-oct28 $E 60 \
    ${2:-sample1.3p1f15g3.5s3,sample1.3p1f15g3.5s4,sample1.3p1f15g3.5s5} 2.5 2>&1 | grep -E "^HELDOUT|rror" | tee $F/e60_$N.log
python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST " | tee -a $F/e60_$N.log
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep -E "MATCH" | tee -a $F/e60_$N.log
echo 0 > $F/$N.exit
