#!/bin/bash
# 2026-11-07: joint Embody-3D + BEAT2 energy distillation (scripts/train_energy_joint.py): privileged energy (3000 steps,
# from the Embody-3D one), distilled causal head (2000 steps), then the BEAT2 test (b2dfwd + 133 ms decoder, energy K=4;
# compare FGD 0.545 without energy, 0.614 with the BEAT2-only head) and Embody-3D UpBeat decoding (energy K=4 +
# adaptive 1.5-5, Db seed 0) on groups 1-7 (compare 0.369 / 3.39 / 2.23 with the Embody-only head).
# usage: CUDA_VISIBLE_DEVICES=0 scripts/energy_joint.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=energy-joint-nov07
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
[ -f outputs/$O/t/energy_teacher.pt ] && grep -q done $F/energy_joint_teacher.log 2>/dev/null || \
    python scripts/train_energy_joint.py --stage teacher --out outputs/$O/t --steps 3000 --init outputs/energy-teacher-nov06/energy_teacher.pt \
    2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/energy_joint_teacher.log
grep -q done $F/energy_joint_head.log 2>/dev/null || \
    python scripts/train_energy_joint.py --stage head --out outputs/$O/h --steps 2000 --energy_teacher outputs/$O/t/energy_teacher.pt \
    2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/energy_joint_head.log
H=outputs/$O/h/energy_head.pt
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5 GEN_EXTRA="--energy 4 --energy_kind d --energy_head $H" \
    scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-nov05 b2dfwd_cf2_e4dJ
M=sample1.3p1f15e4da1.5-5
for g in 1 2 3 4 5 6 7; do
    [ -f $F/dt_UpBeat_joint_grp$g.exit ] || ENERGY_HEAD=$H EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh outputs/dt-Db-nov06/step3000 dt_UpBeat_joint_grp$g $M,${M}s1,${M}s2 outputs/tok-dense-rel-causal-nov06 -2.5
done
echo 0 > $F/energy_joint.exit
