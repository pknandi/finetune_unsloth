#!/bin/bash
# 2026-11-05: repeat of the BEAT2 forward-KL distillation (scripts/beat2_distill.sh, test FGD 0.615 +- 0.038) with
# training seeds 1 and 2, in parallel, each followed by the BEAT2 test, to check the gain is not one lucky run.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_distill_seeds.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
run() {   # <seed>
    O=beat2-dist-fwd-s$1-nov05; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
        --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 --out_dir outputs/$O \
        --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 --seed $1 2>&1 \
        | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && scripts/ol_stream_beat2_test.sh outputs/$O b2dfwds$1
}
run 1 &
run 2 &
wait
echo 0 > $F/beat2_distill_seeds.exit
