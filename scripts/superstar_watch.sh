#!/bin/bash
# 2026-11-02 score every Super Star checkpoint on BEAT2 val as it appears (training keeps only the latest 8,
# one per 10 epochs), oldest first; results appended to outputs/fix-oct21/ss_val.log. Stops when
# outputs/fix-oct21/superstar_watch.stop exists.
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; CD=/data2/home/mubtasim/ext/Super-Star/super_star_beatv2/experiments/lom/Instruct_Mixed_A2M_Causal/checkpoints
while [ ! -f $F/superstar_watch.stop ]; do
    for e in $(ls $CD | sed -n 's/^epoch=\([0-9]*\)\.ckpt$/\1/p' | sort -n); do
        grep -q "^ep$e " $F/ss_val.log && continue
        l=$(scripts/superstar_eval.sh ss_val_ep$e ./experiments/lom/Instruct_Mixed_A2M_Causal/checkpoints/epoch=$e.ckpt val 2>&1 | grep -E "BEAT2|rror" | tail -1)
        echo "ep$e $l" >> $F/ss_val.log
    done
    sleep 600
done
