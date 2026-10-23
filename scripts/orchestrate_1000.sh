#!/bin/bash
# The 2026-10-20 plan for the 1000-row set, end to end. Runs in its own window of tmux session
# `motion`; every training job it starts also gets a `motion` window.
#   1. Capacity tests, tokenizer only, 300 epochs each, in parallel:
#      A = 8 RVQ levels (hidden 1024), B = hidden 2048 (4 levels). Baseline (Q4, hidden 1024): mean ceiling 85.7 at ep400.
#   2. Score both, continue the winner to 1600 epochs, score it again.
#   3. New run dir with the winner's tokenizer, run.sh stages 4-9 (LLM retrain). Retries stage 5+
#      after waiting for a free GPU if it crashes (e.g. another user's job takes the memory).
#   4. Gap check: LLM vs ceiling on BWW760, teacher-forced accuracy, greedy decode.
# Waits for GPU memory before every job (watcher logic). Finished phases leave a marker in $CAP,
# so rerunning this script skips them. Results are appended to docs/EXPERIMENT_LOG.md.
# usage: tmux new-window -t motion -n orch1000 "scripts/orchestrate_1000.sh; exec bash"
set -u
REPO=/home/mubtasim/speech-motion
cd $REPO || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
BASE=outputs/run-overfit1000-tokfix-oct17          # stage-1 CSVs + audio_windows.csv
CAP=$BASE/capacity_oct20
LOG=outputs/terminal/orchestrate_1000.log
DOC=docs/EXPERIMENT_LOG.md
SES=c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600
mkdir -p $CAP

log()  { echo "$(date '+%F %T') $*" | tee -a $LOG; }
note() { echo "| $(date '+%m-%d %H:%M') | $* |" >> $DOC; log "DOC: $*"; }

if ! grep -q '^## Auto-run log' $DOC; then
    printf '\n## Auto-run log (scripts/orchestrate_1000.sh, started 2026-10-20)\n| When | Result |\n|---|---|\n' >> $DOC
fi

# Prints the GPU with the most free memory once it has >= $1 MB on two checks 60 s apart.
wait_gpu() {
    local need=$1 prev="" best bestfree idx free
    while true; do
        best=""; bestfree=0
        while IFS=', ' read -r idx free; do
            if [ "$free" -ge "$need" ] && [ "$free" -gt "$bestfree" ]; then best=$idx; bestfree=$free; fi
        done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
        if [ -n "$best" ] && [ "$best" = "$prev" ]; then echo "$best"; return; fi
        prev=$best; sleep 60
    done
}

# launch_tok <name> <Q> <hidden> <epochs>: tokenizer job in its own motion window
launch_tok() {
    local g; log "waiting for a GPU with 8 GB free for tokenizer $1"; g=$(wait_gpu 8000)
    log "launch tokenizer $1 (Q$2, hidden $3, to epoch $4) on GPU $g -> window motion:tok-$1"
    rm -f $CAP/$1.exit
    tmux new-window -t motion -n "tok-$1" "CUDA_VISIBLE_DEVICES=$g $REPO/scripts/train_tokenizer_job.sh $CAP/$1 $2 $3 $4; exec bash"
}
wait_tok() { while [ ! -f $CAP/$1.exit ]; do sleep 60; done; [ "$(cat $CAP/$1.exit)" = 0 ]; }

# score <name> <tag>: prints "bww760 mean"
score() {
    local g line; g=$(wait_gpu 4000)
    line=$(CUDA_VISIBLE_DEVICES=$g python3 scripts/tokenizer_ceiling.py $CAP/$1 $BASE 30 $CAP/ceiling_$1_$2 2>&1 | tee -a $LOG | grep '^CEILING')
    echo "$line" | sed -E 's/.*bww760=([0-9.na]+) mean=([0-9.na]+).*/\1 \2/'
}

# ---------- 1. capacity tests ----------
if [ ! -f $CAP/phase1.done ]; then
    log "PHASE 1: capacity tests A (Q8) and B (hidden 2048), 300 epochs each"
    launch_tok A_q8 8 1024 300
    sleep 180    # let A's memory show up before picking B's GPU
    launch_tok B_h2048 4 2048 300
    wait_tok A_q8    || { log "A_q8 FAILED, see $CAP/A_q8.log"; exit 1; }
    wait_tok B_h2048 || { log "B_h2048 FAILED, see $CAP/B_h2048.log"; exit 1; }
    touch $CAP/phase1.done
fi

# ---------- 2. pick the winner, train it to 1600 epochs ----------
if [ ! -f $CAP/phase2.done ]; then
    log "PHASE 2: scoring ceilings at epoch 300"
    read -r A_bww A_mean <<< "$(score A_q8 ep300)"
    read -r B_bww B_mean <<< "$(score B_h2048 ep300)"
    note "Capacity test, 300 tokenizer epochs, ceiling BWW760 / mean of 31 rows: **A (Q8, hidden 1024) $A_bww / $A_mean mm**, **B (Q4, hidden 2048) $B_bww / $B_mean mm**. Baseline Q4 / hidden 1024: 122.4 / 85.7 at ep400, 91.1 / 69.8 at ep1600."
    if awk "BEGIN{exit !($A_mean <= $B_mean)}"; then WIN=A_q8; WQ=8; WH=1024; else WIN=B_h2048; WQ=4; WH=2048; fi
    echo "$WIN $WQ $WH" > $CAP/winner
    log "winner: $WIN (Q$WQ, hidden $WH)"
    launch_tok $WIN $WQ $WH 1600
    wait_tok $WIN || { log "$WIN continuation FAILED"; exit 1; }
    read -r W_bww W_mean <<< "$(score $WIN ep1600)"
    echo "$W_bww $W_mean" > $CAP/winner_ceiling
    note "Winner $WIN trained to 1600 epochs: ceiling BWW760 **$W_bww** / mean **$W_mean** mm (baseline at ep1600: 91.1 / 69.8)."
    touch $CAP/phase2.done
fi
read -r WIN WQ WH < $CAP/winner
read -r W_bww W_mean < $CAP/winner_ceiling

# ---------- 3. LLM retrain on the winner's tokens ----------
NEW=run-overfit1000-$( [ "$WQ" = 8 ] && echo q8 || echo h2048 )-oct20
export RUN_NAME=$NEW NUM_QUANTIZERS=$WQ TOK_HIDDEN_DIM_1000=$WH
if [ ! -f $CAP/phase3.done ]; then
    log "PHASE 3: LLM retrain as $NEW (Q$WQ, hidden $WH)"
    if [ ! -e outputs/$NEW ]; then
        mkdir -p /data2/home/mubtasim/speech-motion/outputs/$NEW
        ln -s /data2/home/mubtasim/speech-motion/outputs/$NEW outputs/$NEW
    fi
    mkdir -p outputs/$NEW/motion_tokenizer_artifacts
    cp -n $BASE/tokenizer_dataset_mapping.csv $BASE/tokenizer_dataset_mapping_test.csv \
          $BASE/training_dataset_mapping.csv $BASE/training_dataset_mapping_test.csv \
          $BASE/audio_windows.csv outputs/$NEW/
    cp $CAP/$WIN/tokenizer.pt $CAP/$WIN/normalizer.npz outputs/$NEW/motion_tokenizer_artifacts/
    if [ ! -s outputs/$NEW/speech_motion_train.jsonl ]; then
        g=$(wait_gpu 8000); log "stage 4 (build JSONL) on GPU $g"
        CUDA_VISIBLE_DEVICES=$g STAGE=4 STOP_STAGE=4 ./run.sh || { log "stage 4 FAILED"; exit 1; }
    fi
    FINAL=$(grep -A25 '^        1000)' run.sh | grep -o 'FINAL_CHECKPOINT_STEP=[0-9]*' | head -1 | cut -d= -f2)
    ok=0
    for attempt in 1 2 3 4 5 6; do
        st=5; [ -d outputs/$NEW/lora/checkpoint-$FINAL ] && st=6
        log "attempt $attempt: waiting for a GPU with 32 GB free for stage $st-9"
        g=$(wait_gpu 32000); log "stages $st-9 on GPU $g (attempt $attempt)"
        if CUDA_VISIBLE_DEVICES=$g STAGE=$st ./run.sh; then ok=1; break; fi
        log "run.sh failed (attempt $attempt); retrying after the next GPU wait"
        sleep 300
    done
    [ $ok = 1 ] || { log "stages 5-9 FAILED 6 times"; note "$NEW: stages 5-9 failed 6 times, see $LOG"; exit 1; }
    touch $CAP/phase3.done
fi

# ---------- 4. gap check ----------
if [ ! -f $CAP/phase4.done ]; then
    log "PHASE 4: gap check"
    INF=outputs/$NEW/inference
    METRICS=$(ls $INF/metrics_*.json | head -1)
    LLM=$(python3 -c "import json;print(round(json.load(open('$METRICS'))['root_relative']['mpjpe_mm'],1))")
    START=$(python3 -c "
import csv,os
for r in csv.DictReader(open('$BASE/audio_windows.csv')):
    if '$SES/BWW760/' in r['audio_filename']: print(r['start_sec']); break")
    AUDIO=full_dataset/acting/$SES/BWW760/audio_separated/$SES.wav
    FINAL=$(grep -A25 '^        1000)' run.sh | grep -o 'FINAL_CHECKPOINT_STEP=[0-9]*' | head -1 | cut -d= -f2)
    g=$(wait_gpu 24000)
    TF=$(CUDA_VISIBLE_DEVICES=$g python3 scripts/teacher_forced.py outputs/$NEW/lora/checkpoint-$FINAL outputs/$NEW/speech_motion_train.jsonl $WQ 9 2>&1 | tee -a $LOG | grep '^TEACHER_FORCED')
    g=$(wait_gpu 24000)
    CUDA_VISIBLE_DEVICES=$g python3 speech_to_motion_inference.py --audio_path $AUDIO \
        --lora_model_dir outputs/$NEW/lora/checkpoint-$FINAL \
        --tokenizer_path outputs/$NEW/motion_tokenizer_artifacts/tokenizer.pt \
        --normalizer_path outputs/$NEW/motion_tokenizer_artifacts/normalizer.npz \
        --output_npy_path $INF/greedy_BWW760.npy --num_quantizers $WQ --start_sec $START --greedy 2>&1 | tee -a $LOG
    python3 evaluate_motion.py --session_dir full_dataset/acting/$SES --subject BWW760 --pred_npy $INF/greedy_BWW760.npy \
        --audio_path $AUDIO --smplx_model_dir ./outputs/smplx/models --fps 30 --max_seconds 10.0 \
        --gt_start_sec $START --output_json $INF/metrics_greedy_BWW760.json 2>&1 | tee -a $LOG
    GREEDY=$(python3 -c "import json;print(round(json.load(open('$INF/metrics_greedy_BWW760.json'))['root_relative']['mpjpe_mm'],1))")
    note "$NEW (Q$WQ, hidden $WH, 1600 tok epochs, LLM ckpt $FINAL): BWW760 LLM **$LLM mm** (sampling) / **$GREEDY mm** (greedy) vs tokenizer ceiling **$W_bww** -> gap $(awk "BEGIN{printf \"%.1f\", $LLM-$W_bww}") mm. $TF"
    touch $CAP/phase4.done
fi
log "ALL PHASES DONE"
