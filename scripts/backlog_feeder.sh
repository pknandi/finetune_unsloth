#!/bin/bash
# 2026-11-10: keeps the GPU queue from running dry while nobody checks: every 5 min, if outputs/fix-oct21/gpu_queue.txt is
# empty and a GPU has >= 12 GB free, move the next 4 lines of outputs/fix-oct21/gpu_backlog.txt (same line format) into
# the queue. Stop: touch outputs/fix-oct21/backlog.stop.
# usage: tmux new-window -d -t motion -n feeder "scripts/backlog_feeder.sh; bash"
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; Q=$F/gpu_queue.txt; B=$F/gpu_backlog.txt; touch $B
while [ ! -f $F/backlog.stop ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1 | tr -d ' ')
    if [ ! -s $Q ] && [ -s $B ] && [ "$free" -ge 12000 ]; then
        head -4 $B >> $Q; sed -i '1,4d' $B
        echo "$(date '+%m-%d %H:%M') fed 4 jobs from the backlog ($(wc -l < $B) left)" >> $F/jobs/results.log
    fi
    sleep 300
done
