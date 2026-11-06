#!/bin/bash
# 2026-11-05: weaker reverse-KL weight. kl_w 1 -> speech 0.463 / FD 3.81, kl_w 0.3 -> 0.504 / 3.69 (base 0.378 / 3.43):
# lower weight was better on both, so try kl_w 0.1 (dt-OLd5-nov05). Waits for scripts/distill_fwd.sh (same GPU).
# usage: scripts/distill_kl01.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=dt-OLd5-nov05
until [ -f $F/distill_fwd.exit ]; do sleep 60; done
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
    --batch 24 --kl_w 0.1 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log && scripts/ol_stream_eval.sh outputs/$O/best dt_OLd5_nov05_stream
echo 0 > $F/distill_kl01.exit
