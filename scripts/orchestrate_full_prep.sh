#!/bin/bash
# 2026-10-23: prepare the full dataset (run-full-q8-oct21: 9444 windows) for the aligned model while
# aligned-actpoly-oct23 trains: Q8 / hidden 2048 tokenizer (200 ep), its ceiling, stages 3-4 (motion
# tokens in jsonl), HuBERT features (speech_to_motion_aligned.py --build). GPU jobs wait for memory and
# run in their own `motion` windows. Finished steps leave a marker.
# usage: tmux new-window -t motion -n orchfull "scripts/orchestrate_full_prep.sh; exec bash"
set -u
REPO=/home/mubtasim/speech-motion
cd $REPO || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
FIX=outputs/fix-oct21
LOG=outputs/terminal/orchestrate_full_prep.log
DOC=docs/EXPERIMENT_LOG.md
R=outputs/run-full-q8-oct21
TOK=$R/motion_tokenizer_artifacts
E="env RUN_NAME=run-full-q8-oct21 MAX_SAMPLES="

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

log "START full prep"
if [ ! -f $FIX/fp_tok.done ]; then
    # train_tokenizer_job.sh resumes from $TOK/train_checkpoint.pt; a crash restart continues
    for attempt in 1 2 3; do
        BASE=$R run_job fp_tok 12000 env BASE=$R scripts/train_tokenizer_job.sh $TOK 8 2048 200 && { touch $FIX/fp_tok.done; break; }
        log "fp_tok attempt $attempt failed"
    done
    [ -f $FIX/fp_tok.done ] || { log "fp_tok FAILED"; exit 1; }
fi
if [ ! -f $FIX/fp_ceiling.done ]; then
    run_job fp_ceiling 4000 python3 scripts/tokenizer_ceiling.py $TOK $R 30 $FIX/ceiling_full_h2048 0,1.5 && touch $FIX/fp_ceiling.done
    note "Full-data tokenizer (Q8, hidden 2048, 200 ep, 9444 windows), ceiling over 31 rows: $(grep '^CEILING' $FIX/fp_ceiling.log | sed 's/^CEILING //' | tr '\n' ';'). Acting + polyadic h2048 (450 ep): mean 51.4 / 50.5."
fi
if [ ! -f $FIX/fp_s34.done ]; then
    run_job fp_s34 8000 $E STAGE=3 STOP_STAGE=4 ./run.sh && touch $FIX/fp_s34.done
    [ -f $FIX/fp_s34.done ] || { log "fp_s34 FAILED"; exit 1; }
fi
if [ ! -f $FIX/fp_build.done ]; then
    run_job fp_build 6000 python speech_to_motion_aligned.py --build --run_dir $R --data_dir $R/aligned_data && touch $FIX/fp_build.done
    [ -f $FIX/fp_build.done ] && note "Full data ready for the aligned model: $(wc -l < $R/speech_motion_train.jsonl) train / $(wc -l < $R/speech_motion_val.jsonl) val windows in $R/aligned_data."
fi
log "ALL DONE"
