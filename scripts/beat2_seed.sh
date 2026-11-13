#!/bin/bash
# 2026-11-08: the BEAT2 student's distillation (scripts/beat2_distill.sh b2dfwd: forward KL, 1500 steps) with another
# training seed, then the BEAT2 test with the 133 ms causal decoder, so BEAT2 results get a spread over training seeds
# (seed 0 = beat2-dist-fwd-nov05, test FGD 0.545 mean over 5 draws).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_seed.sh <k>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
K=$1; O=beat2-dist-fwd-s$K-nov08; F=outputs/fix-oct21
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
grep -q "^done" $F/${O}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best \
    --teacher outputs/beat2-ft-f2hw1.0-nov01/best --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 \
    --out_dir outputs/$O --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 --seed $K 2>&1 \
    | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/$O b2dfwd_s${K}_cf2
echo 0 > $F/beat2_seed$K.exit
