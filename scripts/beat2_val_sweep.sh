#!/bin/bash
# 2026-11-03 decoding sweep on the 8 BEAT2 val recordings (never test) for one checkpoint; each setting is
# "<tag>:<generate args>", run 4 at a time. usage: scripts/beat2_val_sweep.sh <name> <ckpt> <tag:args> ...
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; NAME=$1; CK=$2; shift 2
i=0
for s in "$@"; do
    tag=${s%%:*}; a=${s#*:}
    ( python scripts/beat2_generate.py $CK $T $D $O/val_${NAME}_$tag --split val $a 2>&1 | grep --line-buffered rror
      $PY scripts/beat2_score.py $B $R $O/val_${NAME}_$tag --jaw gt --split val 2>&1 | grep -E "BEAT2|rror" | sed "s/^/$tag /" ) &
    i=$((i + 1)); [ $((i % 4)) = 0 ] && wait
done
wait
