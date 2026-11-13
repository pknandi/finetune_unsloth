#!/bin/bash
# 2026-11-07: energy distillation on BEAT2, starting the privileged energy from the Embody-3D one (the BEAT2-only
# energy was weak: val 0.31 vs chance 0.17 on 5.5k windows): privileged energy (3000 steps), then the distilled causal
# head on the BEAT2 student, then the BEAT2 test of b2dfwd + 133 ms decoder with the distilled energy (K=4).
# usage: CUDA_VISIBLE_DEVICES=0 scripts/beat2_energy.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
for O in energy-teacher-b2init-nov07 energy-head-b2-nov07; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
done
A="--student outputs/beat2-dist-fwd-nov05/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best --s_data outputs/beat2-stream-nov04 --t_data outputs/beat2-spk2-oct30/rel-val-hw1.0 --windows 8 --guid 1 --temp 1.0"
python scripts/train_energy_teacher.py $A --out outputs/energy-teacher-b2init-nov07 --steps 3000 \
    --init outputs/energy-teacher-nov06/energy_teacher.pt 2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/energy_teacher_b2init.log
python scripts/train_energy.py $A --out outputs/energy-head-b2-nov07 --steps 1500 --target disc --tau 0.5 \
    --energy_teacher outputs/energy-teacher-b2init-nov07/energy_teacher.pt 2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/energy_head_b2.log
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5 GEN_EXTRA="--energy 4 --energy_kind d --energy_head outputs/energy-head-b2-nov07/energy_head.pt" \
    scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-nov05 b2dfwd_cf2_e4d
echo 0 > $F/beat2_energy.exit
