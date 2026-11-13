#!/bin/bash
# 2026-11-08: full UpBeat decoding for an already distilled Embody-3D student (e.g. a model soup): its own distilled
# energy head (as energy-head-disc-s1-nov07: disc target, Embody privileged energy, 2000 steps), then energy K=4 +
# adaptive 1.5-5 (3 draws) and adaptive only (3 draws) on eval groups 1-7.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/upbeat_from.sh <student ckpt dir> <tag>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
CK=$1; TAG=$2; H=energy-head-$TAG; F=outputs/fix-oct21; T=outputs/tok-dense-rel-causal-nov06
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$H; [ -e outputs/$H ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$H outputs/$H
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student $CK --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 \
    --out outputs/$H --steps 2000 --windows 8 --target disc --energy_teacher outputs/energy-teacher-nov06/energy_teacher.pt --tau 0.5 2>&1 | grep -E "^(VAL|done)|rror" | tee $F/${H}.log && touch $F/${H}.done; }
M=sample1.3p1f15e4da1.5-5; A=sample1.3p1f15a1.5-5
for g in 1 2 3 4 5 6 7; do
    [ -f $F/dt_${TAG}_adapt_grp$g.exit ] || EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh $CK dt_${TAG}_adapt_grp$g $A,${A}s1,${A}s2 $T -2.5
    [ -f $F/dt_UpBeat_${TAG}_grp$g.exit ] || ENERGY_HEAD=outputs/$H/energy_head.pt EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh $CK dt_UpBeat_${TAG}_grp$g $M,${M}s1,${M}s2 $T -2.5
done
echo 0 > $F/upbeat_from_$TAG.exit
