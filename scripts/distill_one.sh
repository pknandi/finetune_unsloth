#!/bin/bash
# 2026-11-05: one Embody distillation run (recipe of dt-OLd2: student dt-OLs24, teacher F2, 24k + 24k streamed windows,
# 3000 steps, batch 24, reverse KL 0.3 unless overridden) + the streamed 60-window eval of best/.
# usage: scripts/distill_one.sh <run_name> [extra distill_onpolicy.py args, e.g. --tip_keep 0.5]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$1; shift
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
    --batch 24 --kl_w 0.3 "$@" 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
scripts/ol_stream_eval.sh outputs/$O/best ${O//-/_}_stream
echo 0 > $F/${O}.exit
