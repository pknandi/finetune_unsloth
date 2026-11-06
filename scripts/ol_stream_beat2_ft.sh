#!/bin/bash
# 2026-11-04: BEAT2 for the online model retrained on online-honest audio. Streams the speaker-2 fine-tune data
# (rel-val-hw1.0, 5,572 train + 52 val windows) once the Embody stream build on this GPU is done, waits for the
# Embody retrain (scripts/ol_stream_train.sh -> dt-OLs-nov04), fine-tunes it on speaker 2 (beat2_ft_hw.sh recipe
# + --causal_audio), picks g1 / g2 at T1.0 on val, then the test bench (--stream), 5 seeded draws + unseeded,
# heading fix, EMAGE scoring -> outputs/fix-oct21/ol_stream_beat2_ft.log.  usage: scripts/ol_stream_beat2_ft.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH STREAM_BS=8
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; SD=outputs/beat2-stream-nov04; r=beat2-ft-olshw1.0-nov04; L=$F/ol_stream_beat2_ft.log
for x in beat2-stream-nov04 $r; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$x; [ -e outputs/$x ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$x outputs/$x
done
until python -c "import numpy as np, sys; sys.exit(0 if np.load('outputs/stream-lower-nov04/done_train.npy').all() else 1)" 2>/dev/null; do sleep 60; done
python scripts/stream_build.py $D/rel-val-hw1.0 $SD --n_train 100000 2>&1 | grep --line-buffered -E "done|rror"
until [ -f $F/ol_stream_train.exit ]; do sleep 60; done
python speech_to_motion_small.py --data_dir $SD --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --init outputs/dt-OLs-nov04/best 2>&1 | grep --line-buffered -v Warn
: > $L
for g in 1 2; do
    v=$O/val_olshw1.0_t1.0g$g
    python scripts/beat2_generate.py outputs/$r/best $T $D $v --split val --temperature 1.0 --guidance $g --stream 2>&1 | grep --line-buffered rror
    $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep -E "BEAT2|rror" | sed "s/^/val g$g /" | tee -a $L
done
g=$(grep "^val g" $L | sed "s/^val g\([0-9]\) .*fgd=\([0-9.]*\).*/\2 \1/" | sort -n | head -1 | cut -d' ' -f2)
echo "val pick g$g" | tee -a $L
N=ft_olshw1.0_t1g$g
scripts/beat2_bench.sh $N outputs/$r/best $T --temperature 1.0 --guidance $g --stream
for d in 3 4; do python scripts/beat2_generate.py outputs/$r/best $T $D $O/${N}_f15s$d --seed_frames 15 --draw $d --temperature 1.0 --guidance $g --stream 2>&1 | grep rror & done; wait
for v in f15s0 f15s1 f15s2 f15s3 f15s4 f0s0; do
    python scripts/beat2_yaw_fix.py $O/${N}_$v $O/${N}_${v}_yaw2a 2 --anchor $B 2>&1 | grep rror
    $PY scripts/beat2_score.py $B $R $O/${N}_${v}_yaw2a --jaw gt 2>&1 | grep -E "BEAT2|rror" | sed "s/^/test $v yaw2a /" | tee -a $L
done
echo 0 > $F/ol_stream_beat2_ft.exit
