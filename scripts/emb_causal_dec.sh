#!/bin/bash
# 2026-11-06: Embody causal-decoder variants (the first zero-look-ahead decoder, tok-dense-rel-causal-nov06, has jerk
# 2.2x real with one-sided smoothing and 11x without; old symmetric decoder 0.4x): zero look-ahead with a 10x
# acceleration loss, and 1-token (67 ms) / 2-token (133 ms) decoder look-ahead; each then the streamed 60-window eval
# of the distilled dt-OLd4-nov05 (3 seeded draws, one-sided smoothing -2.5) for the quality-latency curve.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/emb_causal_dec.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; M3=sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2
for v in "acc10 --acc_w 10" "f1 --future 1" "f2 --future 2"; do
    set -- $v; n=tok-dense-rel-causal-$1-nov06; shift
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$n; [ -e outputs/$n ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$n outputs/$n
    python scripts/train_causal_decoder.py outputs/tok-dense-rel-oct28 outputs/$n \
        --data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --hand_weight 0.3 "$@" 2>&1 \
        | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/cdec_emb_$n.log
done
for v in acc10 f1 f2; do
    scripts/ol_stream_eval_tok.sh outputs/dt-OLd4-nov05/best dt_OLd4_causal_${v}_stream $M3 outputs/tok-dense-rel-causal-$v-nov06 -2.5
done
echo 0 > $F/emb_causal_dec.exit
