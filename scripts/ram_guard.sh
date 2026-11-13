#!/bin/bash
# 2026-11-12: host-RAM guard so the queue can run many evals without OOM. Every 5 s: if MemAvailable < LOW GB (default
# 25), SIGSTOP the newest running aligned_eval.py of ours (its running evaluate_motion.py children finish and free their
# RAM; no new ones start); when MemAvailable > HIGH GB (default 60), SIGCONT the oldest paused one. Never kills anything.
# Paused PIDs: outputs/fix-oct21/ram_guard.paused; log: outputs/fix-oct21/ram_guard.log.
# usage: tmux new-window -t motion -n ramguard "scripts/ram_guard.sh; bash"
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; P=$F/ram_guard.paused; LOW=${LOW:-25}; HIGH=${HIGH:-60}; touch $P
while true; do
    ram=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    # drop paused PIDs that no longer exist
    for p in $(cat $P); do kill -0 $p 2>/dev/null || sed -i "/^$p$/d" $P; done
    if [ "$ram" -lt "$LOW" ]; then
        p=$(ps -u $USER -o pid=,etimes=,stat=,args= | awk '/scripts\/aligned_eval.py/ && $3 !~ /T/ {print $2, $1}' | sort -n | head -1 | awk '{print $2}')
        if [ -n "$p" ]; then kill -STOP $p && echo $p >> $P && echo "$(date '+%m-%d %H:%M:%S') ram=${ram}G pause $p" >> $F/ram_guard.log; fi
    elif [ "$ram" -gt "$HIGH" ] && [ -s $P ]; then
        p=$(head -1 $P); kill -CONT $p 2>/dev/null; sed -i "1d" $P; echo "$(date '+%m-%d %H:%M:%S') ram=${ram}G resume $p" >> $F/ram_guard.log
    fi
    sleep 5
done
