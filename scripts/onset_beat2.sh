#!/bin/bash
# 2026-11-07: speech-onset anticipation on BEAT2 speaker 2: the BEAT2 student's distillation (scripts/beat2_distill.sh,
# b2dfwd: forward KL, 1500 steps) with the onset head (--onset 17 --onset_w 0.5), then the BEAT2 test with the 133 ms
# causal decoder (compare b2dfwd_cf2: FGD 0.545 seed 0, 0.556 over 3 seeds); best/ = lowest val KL, as for b2dfwd.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/onset_beat2.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
O=beat2-dist-fwd-onset-nov07; F=outputs/fix-oct21
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
grep -q "^done" $F/${O}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best \
    --teacher outputs/beat2-ft-f2hw1.0-nov01/best --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 \
    --out_dir outputs/$O --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 --onset 17 --onset_w 0.5 2>&1 \
    | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/$O b2dfwdon_cf2
echo 0 > $F/onset_beat2.exit
