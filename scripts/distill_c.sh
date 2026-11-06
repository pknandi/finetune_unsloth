#!/bin/bash
# 2026-11-06: one Embody run of contributions C/D (guided, deployment-matched on-policy distillation; speech-salient /
# level weights) on the dt-OLd4 recipe (forward KL 1, student dt-OLs24, teacher F2, 3000 steps, batch 24), then the
# streamed 60-window eval of step3000 with the causal decoder (outputs/tok-dense-rel-causal-nov06, one-sided smoothing),
# at the given decoding modes (default seeded T1.3 g3.5 x3).
# usage: scripts/distill_c.sh <run> "<modes or ''>" [extra distill_onpolicy.py args]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$1; MODES=${2:-sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2}; shift 2
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
    --batch 24 --kl_w 0 --fwd_w 1 "$@" 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
scripts/ol_stream_eval_tok.sh outputs/$O/step3000 ${O//-/_}_causal_stream $MODES outputs/tok-dense-rel-causal-nov06 -2.5
echo 0 > $F/${O}.exit
