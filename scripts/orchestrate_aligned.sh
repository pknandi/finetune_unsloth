#!/bin/bash
# 2026-10-23: train and evaluate the time-aligned HuBERT model (speech_to_motion_aligned.py) on the
# acting + polyadic data (run-actpoly-q8-oct21's tokenizer and windows). Each GPU job waits for free
# memory, then runs in its own window of tmux session `motion`. Finished steps leave a marker.
# usage: tmux new-window -t motion -n orchal "scripts/orchestrate_aligned.sh; exec bash"
set -u
REPO=/home/mubtasim/speech-motion
cd $REPO || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
FIX=outputs/fix-oct21
LOG=outputs/terminal/orchestrate_aligned.log
DOC=docs/EXPERIMENT_LOG.md
RUN=outputs/run-actpoly-q8-oct21
DATA=$RUN/aligned_data
NEW=aligned-actpoly-oct23
STEPS=${STEPS:-5400}
if [ ! -e outputs/$NEW ]; then
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$NEW
    ln -s /data2/home/mubtasim/speech-motion/outputs/$NEW outputs/$NEW
fi

log()  { echo "$(date '+%F %T') $*" | tee -a $LOG; }
note() { echo "| $(date '+%m-%d %H:%M') | $* |" >> $DOC; log "DOC: $*"; }
pick_gpu() {
    local need=$1 best="" prev="" bestfree idx free k
    for k in 1 2; do
        prev=$best; best=""; bestfree=0
        while IFS=', ' read -r idx free; do
            if [ "$free" -ge "$need" ] && [ "$free" -gt "$bestfree" ]; then best=$idx; bestfree=$free; fi
        done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
        [ -z "$best" ] && return 1
        [ $k = 1 ] && sleep 60
    done
    [ "$best" = "$prev" ] && echo "$best"
}
launch() {
    local name=$1 need=$2; shift 2
    rm -f $FIX/$name.exit
    log "waiting for a GPU with $need MB free for $name"
    while true; do
        (
            flock 9
            g=$(pick_gpu $need) || exit 1
            [ -n "$g" ] || exit 1
            log "launch $name on GPU $g -> window motion:$name"
            tmux new-window -t motion -n "$name" "cd $REPO; export PATH=$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$g; $* 2>&1 | tee -a $FIX/$name.log; echo \${PIPESTATUS[0]} > $FIX/$name.exit; exec bash"
            sleep 150
        ) 9>$FIX/gpu.lock && return
        sleep 60
    done
}
wait_job() { while [ ! -f $FIX/$1.exit ]; do sleep 60; done; [ "$(cat $FIX/$1.exit)" = 0 ]; }
run_job() { launch "$@"; wait_job $1; }

log "START aligned ($NEW, $STEPS steps)"
if [ ! -f $FIX/al_train.done ]; then
    # resumable only from scratch: a crash restarts training (logs append)
    for attempt in 1 2 3; do
        if run_job al_train 16000 python speech_to_motion_aligned.py --train --data_dir $DATA \
                --out_dir outputs/$NEW --steps $STEPS --eval_every 250; then
            touch $FIX/al_train.done; break
        fi
        log "al_train attempt $attempt failed (see $FIX/al_train.log)"
    done
    [ -f $FIX/al_train.done ] || { log "al_train FAILED 3 times"; exit 1; }
    best=$(grep '^EVAL' $FIX/al_train.log | tail -1)
    note "Aligned HuBERT model ($NEW, acting + polyadic, $STEPS steps), last eval: ${best#EVAL }. Old EnCodec LLM on the same val rows: right 5.10, gap +0.007."
fi
if [ ! -f $FIX/al_eval.done ]; then
    run_job al_eval 14000 python scripts/aligned_eval.py outputs/$NEW/best $DATA $RUN/motion_tokenizer_artifacts \
        $FIX/aligned_eval 10 && touch $FIX/al_eval.done
    note "Aligned model eval (10 held-out clips + BWW760, smoothing 1.5): $(grep -E '^(HELDOUT|TRAIN0)' $FIX/al_eval.log | tr '\n' ';'). Old EnCodec LLM: held-out greedy 290.9, sample 263.2; BWW760 229.3. Videos: $FIX/aligned_eval/"
fi
log "ALL DONE"
