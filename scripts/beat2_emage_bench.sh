#!/bin/bash
# 2026-10-31 BEAT2 scorer check: real motion vs itself, then published EMAGE on the 15 test recordings
# (its paper: FGD 5.512, BC 7.724, L1div 13.06).
cd /home/mubtasim/speech-motion || exit 1
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; R=$PWD/outputs/beat2-spk2-oct30/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21
export CUDA_VISIBLE_DEVICES=${GPU:-1}
$PY scripts/beat2_score.py $B $R gt 2>&1 | grep --line-buffered -E "BEAT2|rror"
$PY scripts/beat2_emage_test.py $B $R $O/emage_pre 2>&1 | grep --line-buffered -E "frames|rror"
for j in keep gt; do $PY scripts/beat2_score.py $B $R $O/emage_pre --jaw $j 2>&1 | grep --line-buffered -E "BEAT2|rror"; done
echo $? > $F/beat2_emage_bench.exit
