#!/bin/bash
# 2026-11-11: 60-window eval of a checkpoint on FULL-WINDOW HuBERT features (outputs/run-full-dense-rel-oct28, the leaky
# input) with a given tokenizer / decoder and smoothing, as scripts/ol_stream_eval_tok.sh does for streamed features
# (audit reference row). EVAL_GROUP=k as in aligned_eval.
# usage: EVAL_GROUP=k scripts/eval_full_tok.sh <ckpt> <name> <modes> <tok_dir> <sigma>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; D=outputs/run-full-q8-oct21; E=$F/e60_$2
NO_VIDEO=1 python scripts/aligned_eval.py $1 outputs/run-full-dense-rel-oct28 $4 $E 60 $3 $5 2>&1 | grep -E "^HELDOUT|rror" | tee $F/e60_$2.log
python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST " | tee -a $F/e60_$2.log
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep MATCH | tee -a $F/e60_$2.log
echo 0 > $F/$2.exit
