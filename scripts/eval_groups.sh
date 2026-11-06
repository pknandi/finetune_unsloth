#!/bin/bash
# 2026-11-06 (D4): a checkpoint on further disjoint groups of 60 Embody val windows (scripts/aligned_eval.py
# EVAL_GROUP), streamed, causal decoder, 3 seeded draws each, so results do not rest on one 60-window set.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/eval_groups.sh <ckpt> <name> [groups="1 2 3 4"] [modes]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
M=${4:-sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2}
for g in ${3:-1 2 3 4}; do
    EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh $1 ${2}_grp$g $M outputs/tok-dense-rel-causal-nov06 -2.5
done
