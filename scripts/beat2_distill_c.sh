#!/bin/bash
# 2026-11-06: contributions C/D on BEAT2: forward-KL distillation of the honest online BEAT2 model (as
# scripts/beat2_distill.sh, b2dfwd = test FGD 0.615) with deployment-matched rollouts (BEAT2 decoding T1.0 g2), the
# teacher's guided distribution as target (2) and level weights, then the BEAT2 test (old decoder, then 1-token
# causal decoder tok-hw1.0-causal-f1).
# usage: CUDA_VISIBLE_DEVICES=0 scripts/beat2_distill_c.sh <run> <tag> [extra distill args]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$1; TAG=$2; shift 2
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
    --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 --out_dir outputs/$O \
    --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 "$@" 2>&1 \
    | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
scripts/ol_stream_beat2_test.sh outputs/$O $TAG
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f1 SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/$O ${TAG}_cf1
echo 0 > $F/$O.exit
