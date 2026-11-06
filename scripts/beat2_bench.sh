#!/bin/bash
# 2026-10-31 BEAT2 benchmark for one of our checkpoints: whole test recordings (scripts/beat2_generate.py),
# 3 draws seeded with the real first 1 s + 1 draw unseeded, each scored with EMAGE's tools (beat2_score.py,
# real jaw / eyes filled in). usage: scripts/beat2_bench.sh <name> <ckpt_dir> <tok_dir> [generate args]
cd /home/mubtasim/speech-motion || exit 1
NAME=$1; CK=$2; TOK=$3; shift 3
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21
V="f15s0 f15s1 f15s2 f0s0"
for v in $V; do     # the 4 variants in parallel (generation is ~6 min per recording each)
    sf=${v:1:2}; sf=${sf%s}; dr=${v##*s}
    python scripts/beat2_generate.py $CK $TOK $D $O/${NAME}_$v --seed_frames $sf --draw $dr "$@" 2>&1 | grep --line-buffered -E "frames|rror" &
done
wait
for v in $V; do $PY scripts/beat2_score.py $B $R $O/${NAME}_$v --jaw gt 2>&1 | grep --line-buffered -E "BEAT2|rror"; done
echo $? > $F/beat2_bench_$NAME.exit
