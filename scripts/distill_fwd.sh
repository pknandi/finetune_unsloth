#!/bin/bash
# 2026-11-05: diversity fix for the on-policy distillation. dt-OLd-nov04 (reverse KL only) gained speech match
# (0.378 -> 0.463) but lost FD (3.43 -> 3.81) and within-clip diversity (1.03 -> 0.87) at every guidance. Mix in
# forward KL(teacher || student) on the student's own rollouts (GKD-style): dt-OLd3 = 0.5 rev + 0.5 fwd,
# then dt-OLd4 = fwd only; each followed by the streamed 60-window eval (seeded g3.5, 3 draws).
# usage: scripts/distill_fwd.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
run() {   # <name> <kl_w> <fwd_w>
    O=$1; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
        --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
        --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
        --batch 24 --kl_w $2 --fwd_w $3 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && scripts/ol_stream_eval.sh outputs/$O/best ${O//-/_}_stream
}
run dt-OLd3-nov05 0.5 0.5
run dt-OLd4-nov05 0 1
echo 0 > $F/distill_fwd.exit
