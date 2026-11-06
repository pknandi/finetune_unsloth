#!/bin/bash
# 2026-10-30: two more independent draws (s1, s2) of the seeded / unseeded guidance-3.5 modes for data tests
# (scripts/data_test.sh), so each setting is scored on 3 samples per clip.  usage: scripts/data_test_repeats.sh <name>...
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
RD=${RD:-outputs/run-full-dense-rel-oct28}; F=outputs/fix-oct21; D=outputs/run-full-q8-oct21
for n in "$@"; do
    until [ -f $F/dt_$n.exit ]; do sleep 60; done
    E=$F/e60_dt_${n}_cfg
    NO_VIDEO=1 python scripts/aligned_eval.py outputs/dt-$n-cfg-oct30/best $RD outputs/tok-dense-rel-oct28 $E 60 \
        sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2,sample1.3p1g3.5s1,sample1.3p1g3.5s2 2.5 2>&1 | grep -E "^HELDOUT" | tee $F/e60_dt_${n}_rep.log
    python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST" | tee -a $F/e60_dt_${n}_rep.log
    python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep MATCH | tee -a $F/e60_dt_${n}_rep.log
    echo 0 > $F/dt_${n}_rep.exit
done
