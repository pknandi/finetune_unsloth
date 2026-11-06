#!/bin/bash
# 2026-10-25: score the small-model variants on 60 held-out clips (sample + static + ceiling, no videos):
# 20-clip, one-sample numbers moved by ~1.5 fd / ~15 mm between reruns, too noisy to rank models.
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH NO_VIDEO=1
D=outputs/run-full-q8-oct21; FIX=outputs/fix-oct21
for e in small:small-full-oct25 la02:small-la02-oct25 la02ad3:small-la02ad3-oct25; do
    ev=${e%%:*}; ck=${e##*:}
    [ "$ev" = la02ad3 ] && while [ ! -f $FIX/small_la02ad3.exit ]; do sleep 60; done
    python scripts/aligned_eval.py outputs/$ck/best $D/aligned_data $D/motion_tokenizer_artifacts $FIX/e60_$ev 60 sample,static,ceiling 2>&1 | grep -E "^(HELDOUT|TRAIN0)" | tee $FIX/e60_$ev.log
    python scripts/motion_dist_metrics.py $FIX/e60_$ev $D/aligned_data 2>&1 | grep ^DIST | tee -a $FIX/e60_$ev.log
done
echo done > $FIX/e60.exit
