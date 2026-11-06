#!/bin/bash
# 2026-11-04: BEAT2 test for the online model on online-honest audio (beat2_generate.py --stream), same decoding as
# the reported OL (beat2-ft-olhw1.0-nov01, T1.0 g1, 3 seeded draws + unseeded), heading fix (_yaw2a), EMAGE scoring.
# Waits for scripts/ol_stream_eval.sh (same GPU).  usage: scripts/ol_stream_beat2.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; R=$PWD/outputs/beat2-spk2-oct30/split/recordings.csv
O=outputs/beat2-bench; F=outputs/fix-oct21; N=ft_olhw1.0_t1g1_stream
until [ -f $F/dt_OL_stream.exit ]; do sleep 60; done
scripts/beat2_bench.sh $N outputs/beat2-ft-olhw1.0-nov01/best outputs/beat2-spk2-oct30/tok-hw1.0 --temperature 1.0 --guidance 1 --stream
for v in f15s0 f15s1 f15s2 f0s0; do
    python scripts/beat2_yaw_fix.py $O/${N}_$v $O/${N}_${v}_yaw2a 2 --anchor $B 2>&1 | grep -E "rror" 
    $PY scripts/beat2_score.py $B $R $O/${N}_${v}_yaw2a --jaw gt 2>&1 | grep -E "BEAT2|rror" | sed "s/^/$v yaw2a /"
done | tee $F/ol_stream_beat2.log
echo 0 > $F/ol_stream_beat2.exit
