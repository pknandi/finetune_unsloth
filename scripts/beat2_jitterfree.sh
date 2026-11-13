#!/bin/bash
# 2026-11-06: jitter-free causal decoders for BEAT2 (as scripts/jitterfree_dec.sh for Embody: acc_w 10, jerk_w 30),
# zero and 2-token look-ahead, then the BEAT2 test of the distilled model (beat2-dist-fwd-nov05) with each.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_jitterfree.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; D=outputs/beat2-spk2-oct30
for v in "jf0 --future 0" "jf2 --future 2"; do
    set -- $v; tag=$1; shift
    python scripts/train_causal_decoder.py $D/tok-hw1.0 $D/tok-hw1.0-causal-$tag --data $D/rel-val-hw1.0 --hand_weight 1.0 \
        --epochs 40 --acc_w 10 --jerk_w 30 "$@" 2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/cdec_b2_$tag.log
    TOK=$D/tok-hw1.0-causal-$tag SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/beat2-dist-fwd-nov05 b2dfwd_$tag
done
echo 0 > $F/beat2_jitterfree.exit
