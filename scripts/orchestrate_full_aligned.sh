#!/bin/bash
# 2026-10-25: train and evaluate the time-aligned HuBERT model (speech_to_motion_aligned.py) on the
# FULL dataset (run-full-q8-oct21: Q8 h2048 tokenizer, 9444 train / 531 val windows), with motion-history
# dropout, LoRA dropout and early stopping (the acting + polyadic run overfit after ~5 epochs). Each GPU job waits for free
# memory, then runs in its own window of tmux session `motion`. Finished steps leave a marker.
# usage: tmux new-window -t motion -n orchfa "scripts/orchestrate_full_aligned.sh; exec bash"
set -u
REPO=/home/mubtasim/speech-motion
cd $REPO || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
FIX=outputs/fix-oct21
LOG=outputs/terminal/orchestrate_full_aligned.log
DOC=docs/EXPERIMENT_LOG.md
RUN=outputs/run-full-q8-oct21
DATA=$RUN/aligned_data
NEW=aligned-full-oct25
STEPS=${STEPS:-12000}
REG="--hist_drop 0.2 --lora_dropout 0.05 --patience 4 --batch 2 --accum 4"
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

log "START full aligned ($NEW, up to $STEPS steps, $REG)"
if [ ! -f $FIX/fa_smoke.done ]; then
    run_job fa_smoke 10000 python speech_to_motion_aligned.py --train --data_dir $DATA --out_dir outputs/$NEW/smoke \
        --steps 20 --eval_every 10 --val_rows 8 --max_rows 16 $REG && touch $FIX/fa_smoke.done
    [ -f $FIX/fa_smoke.done ] || { log "fa_smoke FAILED (see $FIX/fa_smoke.log)"; exit 1; }
    rm -rf outputs/$NEW/smoke
fi
if [ ! -f $FIX/fa_train.done ]; then
    for attempt in 1 2 3; do
        if run_job fa_train 10000 python speech_to_motion_aligned.py --train --data_dir $DATA \
                --out_dir outputs/$NEW --steps $STEPS --eval_every 500 $REG; then
            touch $FIX/fa_train.done; break
        fi
        log "fa_train attempt $attempt failed (see $FIX/fa_train.log)"
    done
    [ -f $FIX/fa_train.done ] || { log "fa_train FAILED 3 times"; exit 1; }
    best=$(grep '^EVAL' $FIX/fa_train.log | sort -t= -k3 -n | head -1)
    note "Aligned HuBERT model on the FULL data ($NEW, $REG), best eval: ${best#EVAL }. Acting + polyadic aligned model: val right 2.893 at step 1750, then overfit."
fi
if [ ! -f $FIX/fa_eval.done ]; then
    run_job fa_eval 12000 python scripts/aligned_eval.py outputs/$NEW/best $DATA $RUN/motion_tokenizer_artifacts \
        $FIX/full_aligned_eval 20 greedy,sample,static,ceiling && touch $FIX/fa_eval.done
    note "Full aligned model eval (20 held-out clips + train row 0, smoothing 1.5): $(grep -E '^(HELDOUT|TRAIN0)' $FIX/fa_eval.log | tr '\n' ';'). Videos: $FIX/full_aligned_eval/"
fi
log "ALL DONE"
