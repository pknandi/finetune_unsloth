#!/bin/bash
# Signal-probe sweep over audio features, with and without motion history (scripts/signal_probe.py).
# usage: CUDA_VISIBLE_DEVICES=<gpu> scripts/run_probes.sh <run_dir> <out_dir>
RUN=$1; OUT=$2
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH OMP_NUM_THREADS=8
mkdir -p "$OUT"
for feat in none encodec8 encodec2 mel hubert; do
  for h in "" "--history"; do
    tag=${feat}${h:+_hist}
    python scripts/signal_probe.py "$RUN" "$OUT" --feat $feat $h > "$OUT/$tag.log" 2>&1
    grep '^PROBE' "$OUT/$tag.log" || echo "PROBE feat=$feat history=${h:+1} FAILED (see $OUT/$tag.log)"
  done
done | tee "$OUT/summary.txt"
echo done > "$OUT/sweep.exit"
