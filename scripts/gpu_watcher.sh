#!/bin/bash
# Waits for a GPU with enough free memory, then launches ./run.sh in a new window of tmux
# session $TMUX_SESSION, pinned to that GPU. Launches exactly once, then exits.
# Usage: setsid nohup scripts/gpu_watcher.sh >/dev/null 2>&1 &     (see docs/OPERATIONS.md)
REPO=/home/mubtasim/speech-motion
CONDA_BIN=/data/home/mubtasim/conda-envs/motion/bin
MIN_FREE_MB=${MIN_FREE_MB:-30000}
POLL_SEC=${POLL_SEC:-60}
TMUX_SESSION=${TMUX_SESSION:-motion}
LOG=$REPO/outputs/terminal/gpu_watcher.log

cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

RUN_NAME=$(grep -m1 '^RUN_NAME=' run.sh | cut -d'"' -f2)
log "watcher start: RUN_NAME=$RUN_NAME MIN_FREE_MB=$MIN_FREE_MB poll=${POLL_SEC}s session=$TMUX_SESSION"

# Fail now, not hours later: the launch window must find the motion env, unsloth and a GPU driver.
if ! "$CONDA_BIN/python" -c "import torch, unsloth, encodec, smplx" >/dev/null 2>&1; then
    log "PREFLIGHT FAILED: motion env imports broken; not watching"; exit 1
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "PREFLIGHT FAILED: tmux session '$TMUX_SESSION' not found; not watching"; exit 1
fi

prev_gpu=""
while true; do
    if pgrep -f "bash ./run.sh" >/dev/null || pgrep -f "speech_to_motion_pipeline.py --train" >/dev/null; then
        sleep "$POLL_SEC"; continue   # one of our runs is already going; don't stack another
    fi
    best_gpu=""; best_free=0
    while IFS=', ' read -r idx free; do
        if [ "$free" -ge "$MIN_FREE_MB" ] && [ "$free" -gt "$best_free" ]; then best_gpu=$idx; best_free=$free; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    if [ -n "$best_gpu" ] && [ "$best_gpu" = "$prev_gpu" ]; then
        log "GPU $best_gpu free (${best_free} MB) on two consecutive checks -> launching $RUN_NAME"
        tmux new-window -t "$TMUX_SESSION" -n "$RUN_NAME" \
            "cd $REPO && export PATH=$CONDA_BIN:\$PATH && export CUDA_VISIBLE_DEVICES=$best_gpu && ./run.sh; echo; echo '[run.sh exited]'; exec bash"
        sleep 20
        if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$RUN_NAME"; then
            log "launched: tmux window $TMUX_SESSION:$RUN_NAME on GPU $best_gpu"
        else
            log "LAUNCH CHECK FAILED: window $RUN_NAME not found in $TMUX_SESSION"
        fi
        exit 0
    fi
    prev_gpu=$best_gpu
    sleep "$POLL_SEC"
done
