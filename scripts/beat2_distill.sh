#!/bin/bash
# 2026-11-05: on-policy distillation on BEAT2 speaker 2 (scripts/distill_onpolicy.py). Student: the honest online BEAT2
# model beat2-ft-olshw1.0-nov04 (test FGD 0.655), teacher: the offline BEAT2 model beat2-ft-f2hw1.0-nov01 (0.575),
# on the streamed BEAT2 windows (outputs/beat2-stream-nov04) vs their full-audio features (rel-val-hw1.0).
# Runs forward-KL-only (best Embody FD) then reverse KL weight 0.3 (best Embody speech match), 1500 steps each,
# each followed by the BEAT2 test (scripts/ol_stream_beat2_test.sh). Waits for scripts/distill_kl01.sh (same GPU).
# usage: scripts/beat2_distill.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
until [ -f $F/distill_kl01.exit ]; do sleep 60; done
run() {   # <name> <tag> <kl_w> <fwd_w>
    O=$1; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
        --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 --out_dir outputs/$O \
        --steps 1500 --batch 24 --kl_w $3 --fwd_w $4 --keep_eval60 --val_rows 52 2>&1 \
        | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && scripts/ol_stream_beat2_test.sh outputs/$O $2
}
run beat2-dist-fwd-nov05 b2dfwd 0 1
run beat2-dist-rev03-nov05 b2drev03 0.3 0
echo 0 > $F/beat2_distill.exit
