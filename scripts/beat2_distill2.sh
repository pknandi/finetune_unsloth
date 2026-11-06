#!/bin/bash
# 2026-11-05: follow-ups to the BEAT2 forward-KL distillation (scripts/beat2_distill.sh, test FGD 0.615 +- 0.038),
# run in parallel on one GPU: forward KL for 3000 steps (b2dfwd3k), and forward KL with TIP Soft-OR keep 0.5
# (b2dfwdtip, 1500 steps). Each is followed by the BEAT2 test (scripts/ol_stream_beat2_test.sh).
# usage: CUDA_VISIBLE_DEVICES=0 scripts/beat2_distill2.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
run() {   # <name> <tag> <steps> [extra]
    O=$1; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
        --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 --out_dir outputs/$O \
        --steps $3 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 "${@:4}" 2>&1 \
        | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && scripts/ol_stream_beat2_test.sh outputs/$O $2
}
run beat2-dist-fwd3k-nov05 b2dfwd3k 3000 &
run beat2-dist-fwdtip-nov05 b2dfwdtip 1500 --tip_keep 0.5 &
wait
echo 0 > $F/beat2_distill2.exit
