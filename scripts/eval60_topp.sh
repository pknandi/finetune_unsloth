#!/bin/bash
# 2026-10-25: re-sample existing small-model checkpoints with top-p 1 (sampling at top-p 0.9 can never
# leave a token whose repeat probability is above 0.9, so confident models freeze) on the same 60 clips.
# usage: scripts/eval60_topp.sh <eval_name> <ckpt_run> <data_dir>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH NO_VIDEO=1
ev=$1; ck=$2; dd=$3; D=outputs/run-full-q8-oct21; FIX=outputs/fix-oct21
python scripts/aligned_eval.py outputs/$ck/best $dd $D/motion_tokenizer_artifacts $FIX/e60_$ev 60 ${MODES:-sample,samplep1,sample1.1p1,static,ceiling} 2>&1 | grep -E "^HELDOUT" | tee $FIX/e60p_$ev.log
python scripts/motion_dist_metrics.py $FIX/e60_$ev $D/aligned_data 2>&1 | grep ^DIST | tee -a $FIX/e60p_$ev.log
echo 0 > $FIX/e60p_$ev.exit
