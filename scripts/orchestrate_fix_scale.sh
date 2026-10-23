#!/bin/bash
# The 2026-10-21 plan: fix at 1000 rows, then scale to acting + polyadic (3370 rows with audio). Runs in its own
# window of tmux session `motion`; every GPU job it starts gets its own `motion` window and waits
# for GPU memory first. Three branches run in parallel:
#   H. held-out check of run-overfit1000-q8-oct20 (ckpt 500 and 2500; greedy / sample / sample_l0)
#   T. acceleration-loss test: Q8 tokenizer at 300 epochs with and without --acc_weight 1, ceilings
#      at smoothing sigma 0 and 1.5. Report only: smoothing already fixed the jitter, so S doesn't wait.
#   S. acting + polyadic run: stages 1-2 (tokenizer); ceiling; stages 3-4; stages 5-9 (retried after GPU
#      waits); then the held-out check on its best checkpoint.
#   A. (added 10-22) audio-shuffle test of the 1000-row model, ckpt 500 and 2500
#   B. (added 10-22) bigger tokenizer: Q8 + hidden 2048 on the acting + polyadic windows, then ceiling
# Smoothing (step 2) was already measured by hand: sigma 1.5 is the run.sh default.
# Finished steps leave a marker in $FIX, so rerunning skips them. Results go to docs/EXPERIMENT_LOG.md.
# usage: tmux new-window -t motion -n orchfix "scripts/orchestrate_fix_scale.sh; exec bash"
set -u
REPO=/home/mubtasim/speech-motion
cd $REPO || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
FIX=outputs/fix-oct21
LOG=outputs/terminal/orchestrate_fix_scale.log
DOC=docs/EXPERIMENT_LOG.md
BASE=outputs/run-overfit1000-tokfix-oct17           # 1000-row stage-1 CSVs + audio_windows.csv
R1000=outputs/run-overfit1000-q8-oct20
NEW=run-actpoly-q8-oct21
for d in fix-oct21 $NEW; do
    if [ ! -e outputs/$d ]; then
        mkdir -p /data2/home/mubtasim/speech-motion/outputs/$d
        ln -s /data2/home/mubtasim/speech-motion/outputs/$d outputs/$d
    fi
done

log()  { echo "$(date '+%F %T') $*" | tee -a $LOG; }
note() { echo "| $(date '+%m-%d %H:%M') | $* |" >> $DOC; log "DOC: $*"; }

# Prints the GPU with the most free memory once it has >= $1 MB on two checks 60 s apart;
# fails (no output) if the two checks disagree or nothing fits.
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

# launch <name> <MB> <command...>: waits for the GPU, runs the command in window motion:<name>,
# output in $FIX/<name>.log, exit code in $FIX/<name>.exit. The lock is held only while one
# check-and-launch runs (plus 150 s for the new job's memory to show up), so a job waiting for
# a big GPU never blocks a smaller one, and two branches never pick the same GPU blindly.
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
run_job() { launch "$@"; wait_job $1; }      # launch and wait

heldout_line() { grep "^HELDOUT mode=$2 " $FIX/$1.log | tail -1 | sed "s/^HELDOUT //"; }

# ---------- H: held-out check of the 1000-row model ----------
branch_H() {
    [ -f $FIX/H.done ] && return
    for ck in 500 2500; do
        launch ho1000_ck$ck 24000 python3 scripts/heldout_eval.py $R1000 $R1000/lora/checkpoint-$ck 8 $FIX/heldout_1000_ck$ck 5
    done
    for ck in 500 2500; do
        wait_job ho1000_ck$ck || { log "ho1000_ck$ck FAILED, see $FIX/ho1000_ck$ck.log"; return 1; }
        note "Held-out check, $R1000 ckpt $ck, 5 unseen acting clips, smoothing 1.5: greedy [$(heldout_line ho1000_ck$ck greedy)]; sample [$(heldout_line ho1000_ck$ck sample)]; sample_l0 [$(heldout_line ho1000_ck$ck sample_l0)]. Videos: $FIX/heldout_1000_ck$ck/"
    done
    touch $FIX/H.done
}

# ---------- T: acceleration-loss test ----------
branch_T() {
    [ -f $FIX/T.done ] && return
    # already launched by an earlier start of this script if their logs exist
    [ -f $FIX/tok_acc0.log ] || launch tok_acc0 8000 scripts/train_tokenizer_job.sh $FIX/tok_acc0 8 1024 300 0
    [ -f $FIX/tok_acc1.log ] || launch tok_acc1 8000 scripts/train_tokenizer_job.sh $FIX/tok_acc1 8 1024 300 1
    wait_job tok_acc0 && wait_job tok_acc1 || { log "acc-loss tokenizer FAILED"; echo 0 > $FIX/acc_weight; return 1; }
    for t in acc0 acc1; do
        run_job ceil_$t 4000 python3 scripts/tokenizer_ceiling.py $FIX/tok_$t $BASE 30 $FIX/ceiling_$t 0,1.5
    done
    local m0 m1 r0 r1
    m0=$(grep '^CEILING.*sigma=1.5 ' $FIX/ceil_acc0.log | sed -E 's/.* mean=([0-9.]+).*/\1/')
    m1=$(grep '^CEILING.*sigma=1.5 ' $FIX/ceil_acc1.log | sed -E 's/.* mean=([0-9.]+).*/\1/')
    r0=$(grep '^CEILING.*sigma=0 ' $FIX/ceil_acc0.log | sed 's/^CEILING //'); r1=$(grep '^CEILING.*sigma=0 ' $FIX/ceil_acc1.log | sed 's/^CEILING //')
    if awk "BEGIN{exit !($m1 <= $m0 - 1.0)}"; then echo 1 > $FIX/acc_weight; else echo 0 > $FIX/acc_weight; fi
    note "Acceleration-loss test (Q8 / hidden 1024, 300 tok epochs, 31 rows). Without: [$r0], mean $m0 at sigma 1.5. With --acc_weight 1: [$r1], mean $m1 at sigma 1.5. Worth adopting (>= 1 mm gain at sigma 1.5): $( [ "$(cat $FIX/acc_weight)" = 1 ] && echo yes || echo no). Report only; the full run uses acc_weight 0."
    touch $FIX/T.done
}

# ---------- S: acting + polyadic ----------
branch_S() {
    local R=outputs/$NEW E="env RUN_NAME=$NEW MAX_SAMPLES=3400 SCENE_TYPES=acting,polyadic"
    if [ ! -s $R/tokenizer_dataset_mapping.csv ]; then
        log "S: stage 1 (CSV) for $NEW"
        $E STAGE=1 STOP_STAGE=1 ./run.sh > $FIX/s1_stage1.log 2>&1 || { log "S: stage 1 FAILED, see $FIX/s1_stage1.log"; return 1; }
    fi
    if [ ! -s $R/audio_windows.csv ]; then      # CPU only, so before the GPU wait
        log "S: window selection for $NEW"
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1) \
            python3 speech_to_motion_pipeline.py --select_windows --csv_path $R/tokenizer_dataset_mapping.csv \
            --output_windows_csv $R/audio_windows.csv.tmp --min_window_rms 0.0002 > $FIX/s_windows.log 2>&1 \
            && mv $R/audio_windows.csv.tmp $R/audio_windows.csv || { log "S: window selection FAILED, see $FIX/s_windows.log"; return 1; }
        log "S: $(tail -1 $FIX/s_windows.log)"
    fi
    if [ ! -f $FIX/s2.exit ] || [ "$(cat $FIX/s2.exit)" != 0 ]; then
        run_job s2 8000 $E STAGE=2 STOP_STAGE=2 ./run.sh || { log "S: stage 2 FAILED, see $FIX/s2.log"; return 1; }
    fi
    if [ ! -f $FIX/s_ceiling.done ]; then
        run_job ceil_full 4000 python3 scripts/tokenizer_ceiling.py $R/motion_tokenizer_artifacts $R 30 $FIX/ceiling_full 0,1.5
        note "$NEW tokenizer (Q8, hidden 1024, 450 ep, acting + polyadic windows), ceiling over BWW760 + 30 training rows: $(grep '^CEILING' $FIX/ceil_full.log | sed 's/^CEILING //' | paste -sd ';'). 1000-row reference: mean 42.1 (sigma 0), 41.0 (sigma 1.5)."
        touch $FIX/s_ceiling.done
    fi
    if [ ! -s $R/speech_motion_train.jsonl ]; then
        run_job s34 8000 $E STAGE=3 STOP_STAGE=4 ./run.sh || { log "S: stages 3-4 FAILED, see $FIX/s34.log"; return 1; }
    fi
    local FINAL=2500 ok=0 st
    for attempt in 1 2 3 4 5 6; do
        [ -f $FIX/s59.done ] && { ok=1; break; }
        st=5; [ -d $R/lora/checkpoint-$FINAL ] && st=6
        # 26 GB: 32 GB sat unmet for 12 h on 10-21/22; the 1000-row run OOM'd with ~18 GB free, and an
        # OOM here just costs a retry from the last checkpoint.
        if run_job s59_a$attempt 26000 $E STAGE=$st ./run.sh; then
            touch $FIX/s59.done; ok=1; break
        fi
        log "S: stages $st-9 failed (attempt $attempt), retrying after the next GPU wait"; sleep 300
    done
    [ $ok = 1 ] || { note "$NEW: stages 5-9 failed 6 times, see $FIX/s59_a*.log"; return 1; }
    local M BEST
    M=$(ls $R/inference/metrics_*.json | head -1)
    BEST=$(python3 -c "import json;print(json.load(open('$R/lora/checkpoint-$FINAL/trainer_state.json')).get('best_model_checkpoint') or '')")
    BEST=$R/lora/$(basename "${BEST:-checkpoint-$FINAL}")
    note "$NEW stages 5-9 done. BWW760 (training clip, greedy, smoothing 1.5): $(python3 -c "import json;d=json.load(open('$M'))['root_relative'];print(f\"{d['mpjpe_mm']:.1f} mm, jerk {d['dynamics']['jerk_ratio_pred_over_gt']:.2f}x\")"). Best held-out-loss checkpoint: $(basename $BEST)."
    if [ ! -f $FIX/s_heldout.done ]; then
        run_job ho_full 24000 python3 scripts/heldout_eval.py $R $BEST 8 $FIX/heldout_full 10 \
            || { log "S: held-out check FAILED"; return 1; }
        note "Held-out check, $NEW $(basename $BEST), 10 unseen clips, smoothing 1.5: greedy [$(heldout_line ho_full greedy)]; sample [$(heldout_line ho_full sample)]; sample_l0 [$(heldout_line ho_full sample_l0)]. Baselines on the 1000-row clips: static mean pose 294 mm, wrong-clip motion 318 mm. Videos: $FIX/heldout_full/"
        touch $FIX/s_heldout.done
    fi
    run_job shuf_actpoly 24000 python3 scripts/audio_shuffle.py $BEST $R/speech_motion_val.jsonl 8 40 \
        || { log "S: audio-shuffle FAILED"; return 1; }
    note "Audio-shuffle test, $NEW $(basename $BEST), 40 held-out rows: $(grep '^AUDIO_SHUFFLE' $FIX/shuf_actpoly.log | sed 's/^AUDIO_SHUFFLE //'). Positive gap = the model uses the audio."
}

# ---------- A: audio-shuffle test of the 1000-row model ----------
branch_A() {
    [ -f $FIX/A.done ] && return
    for ck in 500 2500; do
        run_job shuf1000_ck$ck 24000 python3 scripts/audio_shuffle.py $R1000/lora/checkpoint-$ck $R1000/speech_motion_val.jsonl 8 40 \
            || { log "shuf1000_ck$ck FAILED, see $FIX/shuf1000_ck$ck.log"; return 1; }
        note "Audio-shuffle test, $R1000 ckpt $ck, 40 held-out rows: $(grep '^AUDIO_SHUFFLE' $FIX/shuf1000_ck$ck.log | sed 's/^AUDIO_SHUFFLE //'). Positive gap = the model uses the audio."
    done
    touch $FIX/A.done
}

# ---------- B: Q8 + hidden 2048 tokenizer on acting + polyadic ----------
branch_B() {
    [ -f $FIX/B.done ] && return
    local AP=outputs/run-actpoly-q8-oct21
    [ -f $FIX/tok_h2048_actpoly.log ] || launch tok_h2048_actpoly 12000 env BASE=$AP scripts/train_tokenizer_job.sh $FIX/tok_h2048_actpoly 8 2048 450 0
    wait_job tok_h2048_actpoly || { log "tok_h2048_actpoly FAILED"; return 1; }
    run_job ceil_h2048_actpoly 4000 python3 scripts/tokenizer_ceiling.py $FIX/tok_h2048_actpoly $AP 30 $FIX/ceiling_h2048_actpoly 0,1.5,2
    note "Bigger tokenizer (Q8, hidden 2048, 450 ep, acting + polyadic windows), ceiling over BWW760 + 30 rows: $(grep '^CEILING' $FIX/ceil_h2048_actpoly.log | sed 's/^CEILING //' | paste -sd ';'). Hidden 1024 on the same rows: mean 58.8 (sigma 0), 57.8 (sigma 1.5)."
    touch $FIX/B.done
}

log "START (H, T, S, A, B in parallel)"
branch_H > $FIX/branch_H.out 2>&1 &
branch_T > $FIX/branch_T.out 2>&1 &
branch_S > $FIX/branch_S.out 2>&1 &
branch_A > $FIX/branch_A.out 2>&1 &
branch_B > $FIX/branch_B.out 2>&1 &
wait
log "ALL BRANCHES DONE"
