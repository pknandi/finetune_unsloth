#!/bin/bash
# 2026-11-10: the full UpBeat decoding on BEAT2, so the BEAT2 table has the same rows as Embody (user: reviewers must not
# ask why energy / adaptive guidance are dropped on BEAT2). For each of the 5 distillation runs kept by val FGD
# (RESULTS_FOR_PAPER 2026-11-08 15:05): its own distilled energy head (BEAT2 privileged energy energy-teacher-b2init-nov07,
# as scripts/beat2_energy.sh), then BEAT2 tests at 133 ms with (a) adaptive guidance 1.5-5, (b) energy K=4 at the
# val-picked guidance, (c) energy K=4 + adaptive 1.5-5 (= UpBeat). Compare: the same runs without either 0.570 +- 0.022.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_full_recipe.sh <run dir name> <tag>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; D=outputs/beat2-spk2-oct30; R=$1; TAG=$2; H=energy-head-b2-$TAG-nov10
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$H; [ -e outputs/$H ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$H outputs/$H
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student outputs/$R/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
    --s_data outputs/beat2-stream-nov04 --t_data $D/rel-val-hw1.0 --windows 8 --guid 1 --temp 1.0 --out outputs/$H --steps 1500 \
    --target disc --tau 0.5 --energy_teacher outputs/energy-teacher-b2init-nov07/energy_teacher.pt 2>&1 | grep -E "^(VAL|done)|rror" | tee $F/${H}.log
  [ -f outputs/$H/energy_head.pt ] && touch $F/${H}.done; }
[ -f $F/${H}.done ] || exit 1
ok() { [ $(grep '^test f' $F/ol_stream_beat2_ft_$1.log 2>/dev/null | tail -6 | grep -c 'fgd=') -eq 6 ]; }
export TOK=$D/tok-hw1.0-causal-f2 SIGMA=-2.5
ok ${TAG}_ad || GEN_EXTRA="--adapt 1.5 5" scripts/ol_stream_beat2_test.sh outputs/$R ${TAG}_ad
ok ${TAG}_e4 || GEN_EXTRA="--energy 4 --energy_kind d --energy_head outputs/$H/energy_head.pt" scripts/ol_stream_beat2_test.sh outputs/$R ${TAG}_e4
ok ${TAG}_e4ad || GEN_EXTRA="--energy 4 --energy_kind d --energy_head outputs/$H/energy_head.pt --adapt 1.5 5" scripts/ol_stream_beat2_test.sh outputs/$R ${TAG}_e4ad
ok ${TAG}_ad && ok ${TAG}_e4 && ok ${TAG}_e4ad && echo 0 > $F/beat2_full_$TAG.exit
