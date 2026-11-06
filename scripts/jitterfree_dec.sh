#!/bin/bash
# 2026-11-06 (upgrade 3): jitter-free causal decoders. The zero-look-ahead decoder has jerk 2.2x real (acc_w 10: 1.5x);
# add a jerk term and train zero and 2-token versions, then the Embody eval of Db (step3000) with each.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/jitterfree_dec.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; M3=sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2
for v in "jf0 --acc_w 10 --jerk_w 30" "jf2 --acc_w 10 --jerk_w 30 --future 2"; do
    set -- $v; tag=$1; n=tok-dense-rel-causal-$tag-nov06; shift
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$n; [ -e outputs/$n ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$n outputs/$n
    python scripts/train_causal_decoder.py outputs/tok-dense-rel-oct28 outputs/$n \
        --data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --hand_weight 0.3 "$@" 2>&1 \
        | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/cdec_emb_$n.log
    scripts/ol_stream_eval_tok.sh outputs/dt-Db-nov06/step3000 dt_Db_nov06_${tag}_stream $M3 outputs/$n -2.5
done
echo 0 > $F/jitterfree_dec.exit
