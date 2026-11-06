#!/bin/bash
# 2026-11-06: does the 2-token causal decoder (tok-hw1.0-causal-f2, 133 ms) really beat the old symmetric decoder on
# BEAT2 (b2dfwd: 0.545 vs 0.615)? Same test on the other distillation seeds (old decoder 0.595 / 0.599), the
# undistilled honest online model (old 0.655).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_f2_check.sh
cd /home/mubtasim/speech-motion || exit 1
export TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5
scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-s1-nov05 b2dfwds1_cf2 &
scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-s2-nov05 b2dfwds2_cf2 &
wait
scripts/ol_stream_beat2_test.sh outputs/beat2-ft-olshw1.0-nov04 b2ols_cf2
echo 0 > outputs/fix-oct21/beat2_f2_check.exit
