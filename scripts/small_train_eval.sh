#!/bin/bash
# 2026-10-25: train one speech_to_motion_small.py variant on the dense data, then score it on the 60
# held-out clips (FD against the single-window aligned_data, so numbers match e60_la02ad3 etc.).
# usage: scripts/small_train_eval.sh <name> <out_run_name> <train args...>   (writes $FIX/<name>.exit)
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
name=$1; run=$2; shift 2
D=outputs/run-full-q8-oct21; DD=$D-dense; FIX=outputs/fix-oct21
if [ ! -e outputs/$run ]; then
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$run
    ln -s /data2/home/mubtasim/speech-motion/outputs/$run outputs/$run
fi
python speech_to_motion_small.py --data_dir $DD --out_dir outputs/$run "$@" 2>&1 | tee -a $FIX/${name}_train.log
NO_VIDEO=1 python scripts/aligned_eval.py outputs/$run/best $DD $D/motion_tokenizer_artifacts $FIX/e60_$name 60 ${MODES:-sample1.1p1,sample1.2p1,static,ceiling} ${SIGMA:-1.5} 2>&1 | grep -E "^(HELDOUT|TRAIN0)" | tee $FIX/e60_$name.log
python scripts/motion_dist_metrics.py $FIX/e60_$name $D/aligned_data 2>&1 | grep ^DIST | tee -a $FIX/e60_$name.log
echo 0 > $FIX/$name.exit
