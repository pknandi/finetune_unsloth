#!/bin/bash
# 2026-11-07: speech-onset anticipation on Embody-3D: the UpBeat recipe (Db distillation, METHOD.md arguments, seed 0)
# with the onset head (--onset 17 --onset_w 0.5), its distilled energy head, then UpBeat decoding (energy K=4 + adaptive
# 1.5-5, 3 draws) on groups 1-7. Compare with UpBeat seed 0 (0.369 / 3.39 / 2.23, lag 0 ms) and the 4-seed spread.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/onset_embody.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
R=dt-Don-nov07; H=energy-head-disc-Don-nov07; F=outputs/fix-oct21
[ -d outputs/$R/step3000 ] || scripts/distill_c.sh $R sample1.3p1f15g3.5 --roll_temp 1.3 --roll_guid 3.5 --t_guid 3.5 --lvl_w 2,2,1,1,1,1,0.5,0.5 --onset 17 --onset_w 0.5
[ -d outputs/$R/step3000 ] || { echo "distillation failed"; exit 1; }
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$H; [ -e outputs/$H ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$H outputs/$H
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student outputs/$R/step3000 --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 \
    --out outputs/$H --steps 2000 --windows 8 --target disc --energy_teacher outputs/energy-teacher-nov06/energy_teacher.pt --tau 0.5 2>&1 | grep -E "^(VAL|done)|rror" | tee $F/${H}.log && touch $F/${H}.done; }
M=sample1.3p1f15e4da1.5-5
for g in 1 2 3 4 5 6 7; do
    [ -f $F/dt_UpBeat_onset_grp$g.exit ] || ENERGY_HEAD=outputs/$H/energy_head.pt EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh outputs/$R/step3000 dt_UpBeat_onset_grp$g $M,${M}s1,${M}s2 outputs/tok-dense-rel-causal-nov06 -2.5
done
echo 0 > $F/onset_embody.exit
