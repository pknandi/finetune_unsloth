#!/bin/bash

# Exit immediately if a command crashes
set -e

export CUDA_VISIBLE_DEVICES=0

ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading env from $ENV_FILE"

    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: Env file not found: $ENV_FILE"
fi

STAGE=${STAGE:-1}
STOP_STAGE=${STOP_STAGE:-999}

echo "Running with STAGE=$STAGE STOP_STAGE=$STOP_STAGE"

# ==========================================
# RUN + LOGGING
# ==========================================
RUN_NAME="run-overfit100-sep16"

LOG_DIR="./outputs/terminal"
LOG_FILE="$LOG_DIR/$(date '+%Y-%m-%d_%H-%M-%S').log"

mkdir -p "./outputs/$RUN_NAME"
mkdir -p "$LOG_DIR"
mkdir -p "inference_data/output"

# Save stdout + stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging terminal output to: $LOG_FILE"

CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No color

# ==========================================
# DATA DOWNLOAD CONFIG
# ==========================================

EMBODY3D_SRC="./embody3d.txt"                                          # Download-link list from the Embody-3D release form
EMBODY3D_FEATS="smplx audio text"                                      # Skipping "videos" — unused by this pipeline and by far the largest files
# No --category below on purpose: downloads every category data_download.py knows about.
# Only ~7 of the ~16 extracted scene-type folders (acting, aiagent, daylife, emotions,
# icebreakers, location, polyadic) actually have audio and are usable by this
# audio-conditioned pipeline — the rest (charades, locomotion, hand_*, contact_*,
# *_simon_says) are audio-less per Embody-3D's own feature table, kept only because they
# were downloaded already, and are auto-skipped by dataset_to_csv.py's merged-layout mode.
EMBODY3D_OUTDIR="/data2/home/mubtasim/speech-motion/embody3d_downloads" # Whole-category zips run tens of GB; kept off the small /home filesystem

# ==========================================
# DATASET SELECTION
# ==========================================
# false (default): train on the single-session sample_dataset/ — a memorization sanity
#   check by construction (only one session exists, so train and test necessarily reuse
#   it). Nothing else below changes; this is the exact prior behavior.
# true: train on the full downloaded Embody-3D corpus (full_dataset/ — a symlink to the
#   extracted zips — ~10,200 audio/motion rows across ~3,471 sessions that have audio).
#   TEST_FRAC then holds out whole SESSIONS into a real train/test split, and INFERENCE
#   CONFIG below picks its target from the held-out set instead of a hardcoded session,
#   so this becomes an actual generalization test rather than a memorization check.
USE_FULL_DATASET=true

# Restricts full_dataset training to one scene-type folder (or several, space-separated)
# instead of every scene type with audio — testing the hypothesis that less scenario
# diversity is easier for the model to learn than the full ~7-type mix, given the
# capacity/data constraints found in the RVQ-level diagnostic (see vqvae_motion_tokenizer.py's
# num_quantizers comment). "acting" matches SESSION_NAME's own scenario, so training and
# eval stay in-distribution. Change this one line to try a different scenario or
# combination; "" trains on every scene type with audio (the original full-mix behavior).
# No effect when USE_FULL_DATASET=false (sample_dataset has no scene-type folders).
SCENE_TYPES=""

# ==========================================
# OVERFIT-SCALE SWEEP
# ==========================================
# Bridges the gap between the known-working ~10-sample sample_dataset overfit check
# (USE_FULL_DATASET=false) and the known-broken full ~10,200-row run: caps full_dataset to
# a fixed row count via dataset_to_csv.py's --max_samples, to find the data scale where the
# pipeline starts breaking down. Only takes effect when USE_FULL_DATASET=true.
# "" = no cap (original full-dataset behavior, run-full-sep8's config).
MAX_SAMPLES="100"   # next: change to "1000" (and RUN_NAME/hyperparams below) for the 2nd rung

# A capped subset drawn from every scene type would be a random grab-bag spanning ~7 very
# different scenarios — that confounds "does scale break it" with "does diversity break
# it" (already tested and separated out once: SCENE_TYPES="acting" alone, at ~1219 rows,
# fully memorized in run-acting-sep2 — see MAX_STEPS comment below). So whenever a cap is
# set and SCENE_TYPES wasn't already pinned by hand, default to "acting" (355 sessions /
# 1355 rows available — plenty of session diversity at both 100 and 1000 rows; verified via
# `find full_dataset/acting -path '*/audio_separated/*' -type f | wc -l`).
if [ -n "$MAX_SAMPLES" ] && [ -z "$SCENE_TYPES" ]; then
    SCENE_TYPES="acting"
fi
SCENE_TYPES_FLAG=""
[ -n "$SCENE_TYPES" ] && SCENE_TYPES_FLAG="--scene_types $SCENE_TYPES"

MAX_SAMPLES_FLAG=""
[ -n "$MAX_SAMPLES" ] && MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"

# Everything that differs between the two modes lives in this one branch — split across
# separate variables further down used to mean flipping USE_FULL_DATASET but forgetting
# to also revert TOK_EPOCHS/MAX_STEPS/etc. by hand, silently running the wrong scale's
# hyperparameters against the other dataset. One switch, one place, now.
if [ "$USE_FULL_DATASET" = true ]; then
    DATASET_ROOT="./full_dataset"
    TEST_FRAC=0.1   # fraction of SESSIONS (not rows) held out into a sibling *_test.csv
    # Real run: 5 epochs @ ~130s/epoch, loss 3.13->2.92->2.88->2.86->2.84, drops shrinking
    # ~0.6-0.68x each epoch (geometric decay) — extrapolated total remaining improvement
    # to infinite epochs is only ~0.03 more loss, reached within ~1e-3 by epoch ~13-15.
    # No LR scheduler (fixed 2e-4), so past that is time at a plateau, not convergence.
    TOK_EPOCHS=20
    TOK_BATCH_SIZE=128
    # run-full-aug30 (3000 steps, ~8.7 epochs over ~9,171 rows, num_quantizers=4) had
    # healthy loss (bottomed ~1.79, never near zero) but a teacher-forced accuracy check
    # showed the two finest RVQ levels were barely learned (23%/38% accuracy, ppl 162/51
    # on a 1024-way vocab) — see vqvae_motion_tokenizer.py's num_quantizers comment, now 2.
    #
    # The SCENE_TYPES="acting" test (run-acting-sep2, ~1,219 rows) tried reducing diversity
    # instead/also — refuted: loss went to 0.006 (full memorization of the small training
    # set, not learning), and held-out MPJPE was 1396mm, worse than aug30's 410mm. Each
    # training example there got seen only ~20x (3000 steps x eff.batch 8 / 1219 rows) vs.
    # ~1020x for the original 4-clip memorization test — enough repetition to overfit a
    # small set, not enough to teach it anything transferable. Diversity was helping, not
    # hurting; back to SCENE_TYPES="" (full mix).
    #
    # MAX_STEPS kept at aug30's exact value on purpose — this run's only changed variable
    # vs. aug30 is num_quantizers (4->2), so whatever the next comparison shows attributes
    # cleanly to that fix alone, not to a different training budget too.
    MAX_STEPS=10000
    SAVE_STEPS=500
    FINAL_CHECKPOINT_STEP=10000   # exact multiple of SAVE_STEPS below, so this checkpoint is guaranteed to exist
    DECODE_FLAGS=""   # sampling defaults (do_sample=True) — appropriate once training data is diverse

    # MAX_SAMPLES overrides the block above with configs sized for a much smaller,
    # scenario-coherent subset (see OVERFIT-SCALE SWEEP above). TEST_FRAC stays high enough
    # that even a small capped session pool always holds out >= 1 session (dataset_to_csv.py
    # guarantees this via max(1, round(...))), so eval_loss/MPJPE tracking works the same way
    # it did for run-full-sep3/sep8 — reusing the same diagnostics at each scale is the point.
    # Step budgets target roughly the same per-example exposure that already proved a clean
    # memorization signal at 1219 rows (run-acting-sep2: eff. batch 8 x 3000 steps / 1219
    # rows =~ 19.7 passes, loss bottomed at 0.006) — scaled down for less data to cover per
    # step, not up, since fewer distinct examples should need less total exposure to memorize.
    case "$MAX_SAMPLES" in
        100)
            TEST_FRAC=0.15
            TOK_EPOCHS=300      # tokenizer sees far fewer windows than the full run per
            TOK_BATCH_SIZE=32   # epoch, so needs more epochs to reach the same loss plateau
            MAX_STEPS=800        # eff. batch 8 -> ~64 passes over 100 rows (over-provisioned
            SAVE_STEPS=100        # vs. acting-sep2's ~20, on purpose: this run's whole point
            FINAL_CHECKPOINT_STEP=800  # is to unambiguously see memorization happen)
            ;;
        1000)
            TEST_FRAC=0.1
            TOK_EPOCHS=60
            TOK_BATCH_SIZE=64
            MAX_STEPS=2500      # eff. batch 8 -> ~20 passes over 1000 rows, matching
            SAVE_STEPS=250      # acting-sep2's exposure level at a similar row count
            FINAL_CHECKPOINT_STEP=2500
            ;;
    esac
else
    DATASET_ROOT="./sample_dataset"
    TEST_FRAC=0     # one session total — nothing to hold out
    TOK_EPOCHS=500        # tuned to converge on the 4-clip / ~1,300-window set (original working value)
    TOK_BATCH_SIZE=16
    MAX_STEPS=510         # tuned for the same 4-clip memorization check (original working value)
    SAVE_STEPS=500
    FINAL_CHECKPOINT_STEP=500
    # Greedy reproduces the original memorization-check methodology: the tiny set's
    # "correct" motion-token sequence legitimately repeats the same token 60-90% of the
    # time (long static holds), and sampling/repetition_penalty fights that on purpose —
    # appropriate once training data is diverse (see the true branch), wrong here.
    DECODE_FLAGS="--greedy"
fi

# ==========================================
# TOKENIZER + TRAINING CONFIG
# ==========================================

TOK_ROOT_FOLDER="$DATASET_ROOT"                                       # Tokenizer training dataset
TOK_CSV="./outputs/$RUN_NAME/tokenizer_dataset_mapping.csv"          # audio_filename,motion_dirname
TOK_SAVE_DIR="./outputs/$RUN_NAME/motion_tokenizer_artifacts"        # tokenizer.pkl + normalizer.npz

N_CLUSTERS=1024                                                       # Motion token vocabulary size
# TOK_EPOCHS / TOK_BATCH_SIZE are set above in the USE_FULL_DATASET branch.

TRAIN_ROOT_FOLDER="$DATASET_ROOT"                                     # LLM training dataset
TRAIN_CSV="./outputs/$RUN_NAME/training_dataset_mapping.csv"         # Training dataset CSV (held-out sessions land in a sibling _test.csv when TEST_FRAC > 0)
TRAIN_CSV_TEST="${TRAIN_CSV%.csv}_test.csv"                          # Defined here (not just before STEP 6) so STEP 4 can build a validation JSONL from it too
TRAIN_JSONL="./outputs/$RUN_NAME/speech_motion_train.jsonl"          # Final tokenized dataset
VAL_JSONL="./outputs/$RUN_NAME/speech_motion_val.jsonl"              # Held-out validation JSONL — see STEP 4/5 and finetune()'s eval_dataset comment

OUTPUT_DIR="./outputs/$RUN_NAME/lora"                                # LoRA checkpoints
BASE_MODEL="unsloth/llama-3-8b-bnb-4bit"                             # Base model
OPTIMIZER="muon"                                                     # muon (LoRA 2D matrices on Muon, embed_tokens/lm_head on AdamW) or "adamw" to revert

# MAX_STEPS / SAVE_STEPS are set above in the USE_FULL_DATASET branch (effective batch =
# 1 x grad_accum(8) = 8). finetune() (speech_to_motion_pipeline.py) auto-resumes from the
# latest checkpoint-N in OUTPUT_DIR on its own — no separate resume variable needed here.
LOGGING_STEPS=5

# INFERENCE CONFIG is resolved later, immediately before STEP 6 — see the comment there
# for why (short version: it used to be resolved here, eagerly, before any stage had
# run, which broke a plain end-to-end `./run.sh` the first time it actually mattered).

#################################################################################################################
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 Starting Full Speech-to-Motion Pipeline${NC}"
echo -e "${CYAN}==========================================${NC}"

# ---------------------------------------------------------
# STAGE 0: Download Embody-3D data (opt-in — not part of the 1-9 sequence)
# ---------------------------------------------------------
if [ "$STAGE" -le 0 ] && [ "$STOP_STAGE" -ge 0 ]; then
    echo -e "\n${YELLOW}=> [Stage 0] Downloading all Embody-3D categories (${EMBODY3D_FEATS// /, })...${NC}"

    # STAGE defaults to 1, so this never runs unless you pass STAGE=0
    # explicitly — these are whole-category zips (Scenarios = 49.2h of
    # captures), a one-time slow download, not something to redo every pass.
    # $EMBODY3D_FEATS is deliberately unquoted: it must word-split into
    # separate --feat arguments (argparse nargs="+"), not one string.
    python3 data_download.py \
        --src "$EMBODY3D_SRC" \
        --feat $EMBODY3D_FEATS \
        --outdir "$EMBODY3D_OUTDIR"

    echo -e "\n${GREEN}=> Downloaded to: $EMBODY3D_OUTDIR${NC}"
fi

# ---------------------------------------------------------
# STEP 1: Process the Tokenizer Dataset
# ---------------------------------------------------------
if [ "$STAGE" -le 1 ] && [ "$STOP_STAGE" -ge 1 ]; then
    echo -e "\n${YELLOW}=> [1/9] Generating CSV for Tokenizer Dataset...${NC}"

    python3 dataset_to_csv.py \
        --root_folder $TOK_ROOT_FOLDER \
        --output_csv $TOK_CSV \
        --test_frac $TEST_FRAC \
        $SCENE_TYPES_FLAG \
        $MAX_SAMPLES_FLAG
fi

# ---------------------------------------------------------
# STEP 2: Train Motion Tokenizer
# ---------------------------------------------------------
if [ "$STAGE" -le 2 ] && [ "$STOP_STAGE" -ge 2 ]; then
    echo -e "\n${YELLOW}=> [2/9] Training Motion Tokenizer...${NC}"

    python3 vqvae_motion_tokenizer.py \
        --csv_path $TOK_CSV \
        --save_dir $TOK_SAVE_DIR \
        --n_clusters $N_CLUSTERS \
        --epochs $TOK_EPOCHS \
        --batch_size $TOK_BATCH_SIZE \
        --tokenize_jsonl
fi

# ---------------------------------------------------------
# STEP 3: Process the Fine-Tuning Dataset
# ---------------------------------------------------------
if [ "$STAGE" -le 3 ] && [ "$STOP_STAGE" -ge 3 ]; then
    echo -e "\n${YELLOW}=> [3/9] Generating CSV for Training Dataset...${NC}"

    python3 dataset_to_csv.py \
        --root_folder $TRAIN_ROOT_FOLDER \
        --output_csv $TRAIN_CSV \
        --test_frac $TEST_FRAC \
        $SCENE_TYPES_FLAG \
        $MAX_SAMPLES_FLAG
fi

# ---------------------------------------------------------
# STEP 4: Build Joint Audio-Motion JSONL
# ---------------------------------------------------------
if [ "$STAGE" -le 4 ] && [ "$STOP_STAGE" -ge 4 ]; then
    echo -e "\n${YELLOW}=> [4/9] Building Joint Audio-Motion JSONL for LLM...${NC}"

    python3 speech_to_motion_pipeline.py --build_dataset \
        --csv_path $TRAIN_CSV \
        --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pt" \
        --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
        --output_jsonl $TRAIN_JSONL

    # Held-out validation JSONL, from the same *_test.csv split dataset_to_csv.py already
    # produces when TEST_FRAC > 0 — built here too so STEP 5 can pass it as eval_dataset.
    # Skipped for sample_dataset (TEST_FRAC=0, no split to build from).
    if [ -f "$TRAIN_CSV_TEST" ]; then
        python3 speech_to_motion_pipeline.py --build_dataset \
            --csv_path $TRAIN_CSV_TEST \
            --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pt" \
            --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
            --output_jsonl $VAL_JSONL
    fi
fi

# ---------------------------------------------------------
# STEP 5: Train the LLM
# ---------------------------------------------------------
if [ "$STAGE" -le 5 ] && [ "$STOP_STAGE" -ge 5 ]; then
    echo -e "\n${YELLOW}=> [5/9] Starting Unsloth LoRA Fine-Tuning...${NC}"

    # --val_output_jsonl only passed when the file actually exists (sample_dataset has
    # no test split to build one from) — see finetune()'s eval_dataset comment for why
    # this is the actual fix for "trained 10000 steps and it got worse."
    VAL_JSONL_FLAG=""
    [ -f "$VAL_JSONL" ] && VAL_JSONL_FLAG="--val_output_jsonl $VAL_JSONL"

    python3 speech_to_motion_pipeline.py --train \
        --output_jsonl $TRAIN_JSONL \
        --output_dir $OUTPUT_DIR \
        --base_model $BASE_MODEL \
        --max_steps $MAX_STEPS \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --run_name "$RUN_NAME" \
        --optimizer $OPTIMIZER \
        $VAL_JSONL_FLAG

    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
fi

# ==========================================
# INFERENCE CONFIG
# ==========================================
# Resolved here — right before it's first needed — rather than up top before any stage
# runs. It used to live up top, which broke a single `./run.sh` covering all 9 stages:
# TRAIN_CSV_TEST doesn't exist until STEP 3 creates it, so evaluating this before STEP 1
# even starts always saw "file doesn't exist yet" and left INFERENCE_AUDIO empty — STEP 3
# fixed that on disk hours later, but bash doesn't re-evaluate already-assigned variables,
# so STEP 6 still read the empty value from before STEP 1 ran (`--audio_path` with no
# argument, failing STEP 6 immediately after a full STEP 5 training run completed fine).
# Placed here, unconditionally reached whenever STOP_STAGE >= 6, it runs after STEP 1-5
# have had their chance to execute in *this* invocation, or reads a TRAIN_CSV_TEST left
# by an earlier invocation if this one starts at STAGE=6 — both cases now see the real
# file.
if [ "$STOP_STAGE" -ge 6 ]; then
    if [ "$USE_FULL_DATASET" = true ]; then
        # TRAIN_CSV_TEST is defined earlier now (alongside TRAIN_CSV) so STEP 4 can also use it.
        if [ -f "$TRAIN_CSV_TEST" ]; then
            # Most-dynamic held-out row (by select_inference_sample.py's motion score), not
            # just whichever one happened to land first in the CSV — that used to pick
            # whatever row the shuffle put first regardless of content, which on the
            # 100-sample overfit run landed on a near-static clip with nothing to visually
            # compare against. Still a held-out session/subject the model never trained on,
            # so this stays a real generalization test, the whole point of
            # USE_FULL_DATASET. Layout-agnostic because it reads paths dataset_to_csv.py
            # already resolved, rather than reconstructing them (full_dataset's merged
            # layout has no top-level audio/ or smplx/ split the way sample_dataset's does).
            read -r INFERENCE_AUDIO GT_SUBJECT GT_SESSION_DIR SESSION_NAME < <(python3 select_inference_sample.py --csv_path "$TRAIN_CSV_TEST")
        else
            echo -e "${YELLOW}Warning: $TRAIN_CSV_TEST still doesn't exist (STAGE=3 hasn't run, in this invocation or a prior one) — STAGE>=6 needs it and will fail without it.${NC}"
        fi
    else
        # sample_dataset has exactly one session — this is a memorization sanity check by
        # construction (same session for train and test), not a generalization test.
        SESSION_NAME="c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600"
        # Subject to test — must be one of the ones the model actually saw in training
        # (see outputs/$RUN_NAME/training_dataset_mapping.csv): SZM479, BWW760, DXG448, JON169
        GT_SUBJECT="BWW760"
        GT_SESSION_DIR="${TRAIN_ROOT_FOLDER}/smplx/${SESSION_NAME}"
        INFERENCE_AUDIO="${TRAIN_ROOT_FOLDER}/audio/${SESSION_NAME}/${GT_SUBJECT}/audio_separated/${SESSION_NAME}.wav"
    fi

    AUDIO_FILE_NAME=$(basename "${INFERENCE_AUDIO:-unknown.wav}")
    AUDIO_BASENAME=$(basename "$AUDIO_FILE_NAME" .wav)

    BASE_MODEL_SLUG=$(echo "$BASE_MODEL" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_\|_$//g')

    INFERENCE_CHECKPOINT="./outputs/$RUN_NAME/lora/checkpoint-$FINAL_CHECKPOINT_STEP"

    INFERENCE_OUTPUT="./outputs/$RUN_NAME/inference/generated_motion_${RUN_NAME}_${BASE_MODEL_SLUG}_${GT_SUBJECT}_${AUDIO_BASENAME}.npy"
    INFERENCE_VIDEO="./outputs/$RUN_NAME/inference/video_${RUN_NAME}_${BASE_MODEL_SLUG}_${GT_SUBJECT}_${AUDIO_BASENAME}.mp4"
    COMPARISON_VIDEO="./outputs/$RUN_NAME/inference/comparison_${RUN_NAME}_${BASE_MODEL_SLUG}_${GT_SUBJECT}_${AUDIO_BASENAME}.mp4"
    METRICS_JSON="./outputs/$RUN_NAME/inference/metrics_${RUN_NAME}_${BASE_MODEL_SLUG}_${GT_SUBJECT}_${AUDIO_BASENAME}.json"

    SMPLX_MODEL_DIR="./outputs/smplx/models"
fi

# ---------------------------------------------------------
# STEP 6: Inference
# ---------------------------------------------------------
if [ "$STAGE" -le 6 ] && [ "$STOP_STAGE" -ge 6 ]; then
    echo -e "\n${MAGENTA}=> [6/9] Running Inference on Test Audio...${NC}"

    # $DECODE_FLAGS is deliberately unquoted and may be empty — it's either nothing
    # (sampling defaults) or the single --greedy flag, never a value with spaces.
    python3 speech_to_motion_inference.py\
        --audio_path $INFERENCE_AUDIO \
        --lora_model_dir "$INFERENCE_CHECKPOINT" \
        --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pt" \
        --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
        --base_model $BASE_MODEL \
        --output_npy_path $INFERENCE_OUTPUT \
        $DECODE_FLAGS

    echo -e "${GREEN}Motion output saved to: $INFERENCE_OUTPUT${NC}"
fi

# ---------------------------------------------------------
# STEP 7: Visualization (predicted motion only)
# ---------------------------------------------------------
if [ "$STAGE" -le 7 ] && [ "$STOP_STAGE" -ge 7 ]; then
    echo -e "\n${CYAN}=> [7/9] Rendering SMPL-X Motion Visualization...${NC}"
 
    python3 visualize_motion.py \
        --npy_path        "$INFERENCE_OUTPUT" \
        --smplx_model_dir "$SMPLX_MODEL_DIR" \
        --output_path     "$INFERENCE_VIDEO"
 
    echo -e "\n${GREEN}=> Visualization complete: $INFERENCE_VIDEO${NC}"
fi

# ---------------------------------------------------------
# STEP 8: Side-by-side GT vs Predicted Comparison Video
# ---------------------------------------------------------
if [ "$STAGE" -le 8 ] && [ "$STOP_STAGE" -ge 8 ]; then
    echo -e "\n${CYAN}=> [8/9] Rendering GT vs Predicted Comparison Video...${NC}"

    # Render side-by-side
    python3 compare_motion.py \
        --session_dir "$GT_SESSION_DIR" \
        --subject     "$GT_SUBJECT" \
        --pred_npy    "$INFERENCE_OUTPUT" \
        --audio_path  "$INFERENCE_AUDIO" \
        --output_path "$COMPARISON_VIDEO" \
        --smplx_model_dir "$SMPLX_MODEL_DIR" \
        --fps         30 \
        --max_seconds 10.0 \
        --width       1280 \
        --height      540 \
        --elev        15 \
        --azim        -60

    echo -e "\n${GREEN}=> Comparison video: $COMPARISON_VIDEO${NC}"
fi

# ---------------------------------------------------------
# STEP 9: Quantitative Evaluation (GT vs Predicted)
# ---------------------------------------------------------
if [ "$STAGE" -le 9 ] && [ "$STOP_STAGE" -ge 9 ]; then
    echo -e "\n${CYAN}=> [9/9] Computing Quantitative Metrics (MPJPE, PCK, Beat Consistency)...${NC}"

    # Same GT/pred loading + frame alignment as compare_motion.py's video, so
    # these numbers score exactly what that comparison video shows.
    python3 evaluate_motion.py \
        --session_dir "$GT_SESSION_DIR" \
        --subject     "$GT_SUBJECT" \
        --pred_npy    "$INFERENCE_OUTPUT" \
        --audio_path  "$INFERENCE_AUDIO" \
        --smplx_model_dir "$SMPLX_MODEL_DIR" \
        --fps         30 \
        --max_seconds 10.0 \
        --output_json "$METRICS_JSON"

    echo -e "\n${GREEN}=> Metrics saved: $METRICS_JSON${NC}"
fi
