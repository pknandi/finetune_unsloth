#!/bin/bash
# 2026-11-04: the strict online model (dt-OL-cfg-oct30) on online-honest audio features (scripts/stream_hubert.py:
# each slot from the audio up to its end only), Embody-3D 60 eval windows, seeded g3.5, 3 draws, as e60_dt_OL_rep.
# usage: scripts/ol_stream_eval.sh [ckpt=outputs/dt-OL-cfg-oct30/best] [name=dt_OL_stream] [modes=seeded g3.5 x3]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
CK=${1:-outputs/dt-OL-cfg-oct30/best}; N=${2:-dt_OL_stream}
RD=outputs/run-full-dense-rel-oct28; SD=outputs/stream-hubert-nov04; F=outputs/fix-oct21; D=outputs/run-full-q8-oct21
mkdir -p /data2/home/mubtasim/speech-motion/$SD; [ -e $SD ] || ln -s /data2/home/mubtasim/speech-motion/$SD $SD
[ -f $SD/val_hubert.npy ] && [ -e $SD/meta.json ] || python scripts/stream_hubert.py $RD $SD 2>&1 | grep -E "^(streamed|wrote)|rror"
E=$F/e60_$N
NO_VIDEO=1 python scripts/aligned_eval.py $CK $SD outputs/tok-dense-rel-oct28 $E 60 \
    ${3:-sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2} 2.5 2>&1 | grep -E "^HELDOUT|rror" | tee $F/e60_$N.log
python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST " | tee -a $F/e60_$N.log
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep -E "MATCH" | tee -a $F/e60_$N.log
echo 0 > $F/$N.exit
