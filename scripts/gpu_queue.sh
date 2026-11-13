#!/bin/bash
# 2026-11-07: GPU job queue so the GPUs never sit idle between checks. Every 60 s, for each GPU (0, 1) with at least
# NEED_MB free (default 12000) and fewer than MAX_JOBS (default 2) jobs of ours started by this queue, pop the first
# line of outputs/fix-oct21/gpu_queue.txt and run it in its own tmux window of session motion with
# CUDA_VISIBLE_DEVICES set (each line: <window name> <command...>; the command runs from the repo with the motion env).
# Each job gets 8 CPU threads (OMP / MKL / OpenBLAS): the evals are CPU-heavy and the box has 96 cores shared.
# Lines are removed when started; started jobs are appended to gpu_queue.started. Stop: touch gpu_queue.stop.
# 2026-11-08: each job runs from its own file (outputs/fix-oct21/jobs/<name>.sh) through scripts/run_job2.sh, which
# re-queues a failed job (non-zero exit, or its CHECK command fails) up to 2 times; see scripts/run_job.sh.
# usage: tmux new-window -t motion -n gpuq "scripts/gpu_queue.sh; bash"
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; Q=$F/gpu_queue.txt; MAXJ=${MAX_JOBS:-999}; MARGIN=${MARGIN_MB:-2000}
# 2026-11-08: no job cap by default; each job starts only when its GPU has the job's own memory need free plus MARGIN_MB:
# a "#MEM# <MB>" tag on the queue line, else 6000 for Embody evals (ol_stream_eval_tok.sh: ~3 x 1.6 GB), 8000 for BEAT2
# tests, 12000 for anything else (training, data builds)
need_of() { case "$1" in *" #MEM# "*) echo "$1" | sed 's/.* #MEM# \([0-9]*\).*/\1/';; *ol_stream_eval_tok.sh*) echo 6000;; *ol_stream_beat2_test.sh*) echo 8000;; *) echo 12000;; esac; }
touch $Q; mkdir -p $F/jobs
while [ ! -f $F/gpu_queue.stop ]; do
    for g in 0 1; do
        [ -s $Q ] || break
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d ' ')
        running=$(tmux list-windows -t motion -F '#W' | grep -c "^q${g}_")
        line=$(head -1 $Q); NEED=$(( $(need_of "$line") + MARGIN ))
        if [ "$free" -ge "$NEED" ] && [ "$running" -lt "$MAXJ" ]; then
            sed -i '1d' $Q; line=$(echo "$line" | sed 's/ #MEM# [0-9]*//')
            name=$(echo "$line" | cut -d' ' -f1); cmd=$(echo "$line" | cut -d' ' -f2-)
            case "$cmd" in *" #CHECK# "*) printf '%s\nCHECK=%s\n' "${cmd%% #CHECK# *}" "${cmd#* #CHECK# }";; *) printf '%s\n' "$cmd";; esac > $F/jobs/$name.sh
            tmux new-window -d -t motion -n "q${g}_$name" "cd /home/mubtasim/speech-motion; export PATH=/data/home/mubtasim/conda-envs/motion/bin:\$PATH; CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 scripts/run_job2.sh $name; tmux rename-window done_$name"
            echo "$(date '+%m-%d %H:%M') gpu$g $line" >> $F/gpu_queue.started
            sleep 45                                     # let it allocate before the next check of this GPU
        fi
    done
    sleep 30
done
