#!/bin/bash
# 2026-11-06: BEAT2 causal-decoder variants for the quality-latency curve (the first zero-look-ahead decoder lost
# test FGD 0.615 -> 0.781 on beat2-dist-fwd-nov05): zero look-ahead wider (512) and longer; 1 token (67 ms) and
# 2 tokens (133 ms) of decoder look-ahead. Each decoder, then the BEAT2 test of beat2-dist-fwd-nov05 with it
# (one-sided smoothing sigma -2.5), and the first decoder also without smoothing.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_causal_dec.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; D=outputs/beat2-spk2-oct30
dec() {   # <name> <args>
    python scripts/train_causal_decoder.py $D/tok-hw1.0 $D/$1 --data $D/rel-val-hw1.0 --hand_weight 1.0 "${@:2}" 2>&1 \
        | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/cdec_b2_$1.log
}
dec tok-hw1.0-causal-w512 --epochs 80 --dec_hidden 512
dec tok-hw1.0-causal-f1 --epochs 40 --future 1
dec tok-hw1.0-causal-f2 --epochs 40 --future 2
TOK=$D/tok-hw1.0-causal-nov06 SIGMA=0 scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-nov05 b2dfwd_causal_s0
for v in w512 f1 f2; do
    TOK=$D/tok-hw1.0-causal-$v SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-nov05 b2dfwd_causal_$v
done
echo 0 > $F/beat2_causal_dec.exit
