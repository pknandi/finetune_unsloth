#!/bin/bash
# 2026-11-04: NLL-only control for scripts/distill_run.sh: same student init, data, steps, batch, schedule and
# --nll_w, but --kl_w 0 (no teacher). If dt-OLd beats this, the gain is the distillation, not extra fine-tuning.
# usage: scripts/distill_control.sh [steps=3000]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=dt-OLn-nov04
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/leak_probe.py outputs/dt-OLs24-la13-nov04/best stream 2>&1 | grep -E 'LEAK|rror' | tee $F/leak_la13.log
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps ${1:-3000} \
    --batch 24 --kl_w 0 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/distill_control_train.log
grep -q "^done" $F/distill_control_train.log || exit 1
scripts/ol_stream_eval.sh outputs/$O/best dt_OLn_stream
echo 0 > $F/distill_control.exit
