#!/bin/bash
# 2026-11-07: the whole UpBeat recipe for one more training seed: Db distillation (scripts/distill_c.sh with the
# METHOD.md arguments, --seed k), its distilled energy head (as energy-head-disc-s1-nov07), then UpBeat decoding
# (e4da1.5-5, 3 draws) on the fresh eval groups 1-7. Outputs: outputs/dt-Db-s<k>-nov07, outputs/energy-head-disc-s<k>-nov07,
# outputs/fix-oct21/e60_dt_UpBeat_s<k>_grp<g>.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/upbeat_seed.sh <k>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
K=$1; R=dt-Db-s$K-nov07; H=energy-head-disc-s$K-nov07; F=outputs/fix-oct21
[ -f outputs/$R/step3000/model.pt ] || [ -d outputs/$R/step3000 ] || scripts/distill_c.sh $R sample1.3p1f15g3.5 --roll_temp 1.3 --roll_guid 3.5 --t_guid 3.5 --lvl_w 2,2,1,1,1,1,0.5,0.5 --seed $K
[ -d outputs/$R/step3000 ] || { echo "distillation failed"; exit 1; }
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$H; [ -e outputs/$H ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$H outputs/$H
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student outputs/$R/step3000 --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 \
    --out outputs/$H --steps 2000 --windows 8 --target disc --energy_teacher outputs/energy-teacher-nov06/energy_teacher.pt --tau 0.5 2>&1 | grep -E "^(VAL|done)|rror" | tee $F/${H}.log && touch $F/${H}.done; }
M=sample1.3p1f15e4da1.5-5
for g in 1 2 3 4 5 6 7; do
    [ -f $F/dt_UpBeat_s${K}_grp$g.exit ] || ENERGY_HEAD=outputs/$H/energy_head.pt EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh outputs/$R/step3000 dt_UpBeat_s${K}_grp$g $M,${M}s1,${M}s2 outputs/tok-dense-rel-causal-nov06 -2.5
done
echo 0 > $F/upbeat_seed$K.exit
