#!/bin/bash
# 2026-11-04: as scripts/ol_stream_beat2_ft.sh (BEAT2 speaker-2 fine-tune of an online-honest model on the streamed
# BEAT2 data, val pick g1 / g2 at T1.0, test 5 seeded draws + unseeded with heading fix), from a given Embody init,
# generators 2 at a time (4 in parallel OOMed next to another user's job on GPU 0).
# usage: scripts/ol_stream_beat2_ft2.sh <init_ckpt> <tag>     e.g. outputs/dt-OLs24-nov04/best ols24
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH STREAM_BS=8
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; SD=outputs/beat2-stream-nov04; INIT=$1; TAG=$2
r=beat2-ft-${TAG}hw1.0-nov04; L=$F/ol_stream_beat2_ft_$TAG.log
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $SD --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --init $INIT 2>&1 | grep --line-buffered -v Warn
[ -f outputs/$r/best/small.pt ] || { echo "fine-tune failed (no outputs/$r/best)" | tee -a $L; exit 1; }
: > $L
for g in 1 2; do
    v=$O/val_${TAG}hw1.0_t1.0g$g
    python scripts/beat2_generate.py outputs/$r/best $T $D $v --split val --temperature 1.0 --guidance $g --stream 2>&1 | grep --line-buffered rror &
done; wait
for g in 1 2; do $PY scripts/beat2_score.py $B $R $O/val_${TAG}hw1.0_t1.0g$g --jaw gt --split val 2>&1 | grep -E "BEAT2|rror" | sed "s/^/val g$g /" | tee -a $L; done
g=$(grep "^val g" $L | sed "s/^val g\([0-9]\) .*fgd=\([0-9.]*\).*/\2 \1/" | sort -n | head -1 | cut -d' ' -f2)
echo "val pick g$g" | tee -a $L
N=ft_${TAG}hw1.0_t1g$g
gen() { python scripts/beat2_generate.py outputs/$r/best $T $D $O/${N}_f$1s$2 --seed_frames $1 --draw $2 --temperature 1.0 --guidance $g --stream 2>&1 | grep rror; }
gen 15 0 & gen 15 1 & wait; gen 15 2 & gen 15 3 & wait; gen 15 4 & gen 0 0 & wait
for v in f15s0 f15s1 f15s2 f15s3 f15s4 f0s0; do
    python scripts/beat2_yaw_fix.py $O/${N}_$v $O/${N}_${v}_yaw2a 2 --anchor $B 2>&1 | grep rror
    $PY scripts/beat2_score.py $B $R $O/${N}_${v}_yaw2a --jaw gt 2>&1 | grep -E "BEAT2|rror" | sed "s/^/test $v yaw2a /" | tee -a $L
done
echo 0 > $F/ol_stream_beat2_ft_$TAG.exit
