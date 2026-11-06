#!/bin/bash
# 2026-11-05: fixes for the distillation's realism / diversity cost (reverse KL weight 0.3: speech 0.504, FD 3.69 vs
# base 0.404 / 3.43), from the papers the user shared: (a) TIP token selection (--tip_keep 0.5: KL only on uncertain or
# confident-but-wrong tokens), (b) a softer teacher (--t_temp 1.3, the temperature we sample at). Same recipe as
# dt-OLd2 otherwise; each + the streamed 60-window eval. Waits for scripts/beat2_distill.sh (same GPU).
# usage: scripts/distill_v2.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
until [ -s $F/beat2_distill.exit ]; do sleep 60; done
run() {   # <name> <extra args...>
    O=$1; shift
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
        --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
        --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
        --batch 24 --kl_w 0.3 "$@" 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && scripts/ol_stream_eval.sh outputs/$O/best ${O//-/_}_stream
}
run dt-OLd6-nov05 --tip_keep 0.5
run dt-OLd7-nov05 --t_temp 1.3
echo 0 > $F/distill_v2.exit
