#!/bin/bash
# 2026-11-04: dt-OLd-nov04 (on-policy distillation) gained speech match (0.378 -> 0.463) but lost FD (3.43 -> 3.81) and
# within-clip diversity (1.03 -> 0.87): reverse KL is mode-seeking, and g3.5 guidance sharpens further. Checks:
#   (a) dt-OLd best at lower guidance g2.0 / g2.5;   (b) dt-OLd step1000 at g3.5 (less distilled);
#   (c) dt-OLd2-nov04: distillation with --kl_w 0.3 (same everything else), then the g3.5 eval.
# (a)+(b) run next to (c) on the same GPU.   usage: scripts/distill_followup.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=dt-OLd2-nov04
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
(python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps 3000 \
    --batch 24 --kl_w 0.3 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/distill2_train.log
 grep -q "^done" $F/distill2_train.log && scripts/ol_stream_eval.sh outputs/$O/best dt_OLd2_stream) &
scripts/ol_stream_eval.sh outputs/dt-OLd-nov04/best dt_OLd_g2 sample1.3p1f15g2.0,sample1.3p1f15g2.0s1,sample1.3p1f15g2.0s2
scripts/ol_stream_eval.sh outputs/dt-OLd-nov04/best dt_OLd_g25 sample1.3p1f15g2.5,sample1.3p1f15g2.5s1,sample1.3p1f15g2.5s2
scripts/ol_stream_eval.sh outputs/dt-OLd-nov04/step1000 dt_OLd_s1000
wait
echo 0 > $F/distill_followup.exit
