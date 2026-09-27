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
RUN_NAME="run-overfit100-tokfix-sep27"

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
MAX_SAMPLES="100"

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
[ -n "$MAX_SAMPLES" ] && MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES --force_session c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600"
# --force_session guarantees the fixed inference clip below (BWW760) is always actually IN
# the training set, regardless of the random sample -- see dataset_to_csv.py's comment.

# RVQ levels per 2-frame motion block — must be the same in the tokenizer (STEP 2), the LLM
# (STEP 5) and inference (STEP 6); passed to all three below. 2 is the default. Measured
# reconstruction ceiling (GT tokens -> decode, same seed): 4 levels beat 2 by ~35-40% at
# both 4-clip and 100-row scale, but the LLM couldn't learn the fine levels on the full
# corpus, so 4 is for the memorization/overfit checks unless that changes.
NUM_QUANTIZERS=4    # Tokenizer-only ceiling, same 26 windows, all other settings equal (root-relative
                    # MPJPE mean/BWW760, mm): Q2 42.0/76.6, Q3 29.9/43.2, Q4 25.0/41.8. Each residual
                    # level adds a tier of detail; more codes per level does not (see N_CLUSTERS).

# Minimum RMS a row's best select_best_audio_window() window must clear to be kept in
# STEP 4's JSONL — 0.0 (default) keeps everything, matching the original behavior before
# per-row audio windowing existed. Only the "25)" case below overrides this; see its
# comment for why 0.0002 specifically.
MIN_WINDOW_RMS=0.0

# true: pick each row's 10 s window BEFORE the tokenizer (STEP 2) and train the tokenizer on
# those windows only, instead of whole clips; STEP 4 then builds the LLM examples from the same
# windows file, so the two can't disagree. Whole-clip training gave the tokenizer 12x more motion
# than the LLM ever sees (3180 s vs 260 s on the 25-row set) and its capacity went to frames that
# are never scored: reconstruction error on the used windows fell from 114.7 to 44.8 mm
# (root-relative MPJPE, same RVQ depth) just by restricting training to them. This is the right
# setting for a memorization run; for a generalization run whole clips give broader coverage
# (untested), so it stays false outside the "25)" case.
TOK_ON_WINDOWS=false

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
        25)
            # Memorization check on 25 rows (BWW760's session forced in). Everything that made the
            # earlier 25-row runs miss overfit4 was in the tokenizer, not the LLM: the LLM's output
            # matched the tokenizer's own reconstruction ceiling to the millimetre every time. The
            # tokenizer-only fixes (each measured, see the flags/comments they live next to):
            #   - 6D rotations for root/body/hands (sin/cos on the root orient decodes to a different
            #     rotation once its norm passes pi: +25 mm on the BWW760 target, +253 mm on GBM952)
            #   - tokenizer trained on the LLM's own windows (TOK_ON_WINDOWS)
            #   - finger channels weighted 0.3 (55% of input channels, 0% of the metric's joints)
            # Ceiling on the 26 kept rows went 114.7 -> ~24 mm mean; BWW760 112 -> ~36-43 mm.
            # overfit4's real LLM output, scored the same way, is 60.3 mm.
            TEST_FRAC=0
            TOK_EPOCHS=2000         # windows-only set is ~270 chunks/epoch (17 steps at batch 16)
            TOK_BATCH_SIZE=16
            TOK_ON_WINDOWS=true
            MAX_STEPS=3200          # 1020 passes/row (overfit4's exposure): 1020*25/8 = 3187.5
            SAVE_STEPS=400
            FINAL_CHECKPOINT_STEP=3200
            DECODE_FLAGS=""
            # 0.0002: a clip whose loudest 10 s window is still under this is true silence
            # throughout; reclipping cannot help (collapsed pairs sharing >50% identical audio
            # tokens 143->24, >90% 20->1, on the other 26 rows).
            MIN_WINDOW_RMS=0.0002
            ;;
        100)
            # Same tokenizer-only fixes validated at 25 rows (see that case's comment): 6D
            # rotations, window-matched tokenizer training, finger weight 0.3 -- all apply here
            # automatically (they're defaults, not scale-specific flags). Measured directly at
            # this scale though (81 kept rows, same architecture/codebook): ceiling is real but
            # meaningfully higher than at 25 rows -- mean 52.4mm / BWW760 63.6mm at 2000 epochs,
            # vs ~25mm / ~42mm at 25 rows -- because 4x the distinct motion now shares the same
            # 1024x4 codebook and 256-wide encoder/decoder. Loss plateaus noisily around 0.80-
            # 0.85 well before 2000 epochs (1200ep: mean 56.9/BWW760 98.8; 2000ep: 52.4/63.6), so
            # more epochs alone won't close this further -- untried next step would be more
            # codebook/model capacity, which also needs the LLM's <m_q_i> vocabulary (currently
            # 1024, see MOTION_VOCAB_SIZE) raised to match. Not attempted here; this config is
            # "apply what's proven," not a new capacity fix.
            TEST_FRAC=0.15
            TOK_EPOCHS=2000
            TOK_BATCH_SIZE=32
            TOK_ON_WINDOWS=true
            MIN_WINDOW_RMS=0.0002
            MAX_STEPS=800        # eff. batch 8 -> ~64 passes over 100 rows; teacher-forced
            SAVE_STEPS=100        # accuracy was already 99.8-100% at this budget pre-fix, so
            FINAL_CHECKPOINT_STEP=800  # LLM capacity was never the bottleneck here, only the tokenizer was.
            ;;
        1000)
            MIN_WINDOW_RMS=0.0
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

N_CLUSTERS=1024                                                       # must equal the LLM's <m_q_i> vocabulary (MOTION_VOCAB_SIZE in
                                                                      # speech_to_motion_pipeline.py; STEP 4 refuses anything else).
                                                                      # 2048 codes gave the same ceiling as 1024 (24.7 vs 24.1 mm mean).
# TOK_EPOCHS / TOK_BATCH_SIZE are set above in the USE_FULL_DATASET branch.

TRAIN_ROOT_FOLDER="$DATASET_ROOT"                                     # LLM training dataset
TRAIN_CSV="./outputs/$RUN_NAME/training_dataset_mapping.csv"         # Training dataset CSV (held-out sessions land in a sibling _test.csv when TEST_FRAC > 0)
TRAIN_CSV_TEST="${TRAIN_CSV%.csv}_test.csv"                          # Defined here (not just before STEP 6) so STEP 4 can build a validation JSONL from it too
TRAIN_JSONL="./outputs/$RUN_NAME/speech_motion_train.jsonl"          # Final tokenized dataset
WINDOWS_CSV="./outputs/$RUN_NAME/audio_windows.csv"                  # Per-row 10 s windows, written by STEP 2 when TOK_ON_WINDOWS=true
TOK_WINDOWS_FLAG=""
TRAIN_WINDOWS_FLAG=""
if [ "$TOK_ON_WINDOWS" = true ]; then
    TOK_WINDOWS_FLAG="--windows_csv $WINDOWS_CSV"
    TRAIN_WINDOWS_FLAG="--windows_csv $WINDOWS_CSV"
fi
VAL_JSONL="./outputs/$RUN_NAME/speech_motion_val.jsonl"              # Held-out validation JSONL — see STEP 4/5 and finetune()'s eval_dataset comment

OUTPUT_DIR="./outputs/$RUN_NAME/lora"                                # LoRA checkpoints
BASE_MODEL="unsloth/llama-3-8b-bnb-4bit"                             # Base model
OPTIMIZER="adamw"                                                    # muon has a known lr-scale defect (see _step_muon) — adamw is the setting that overfit before

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

    if [ "$TOK_ON_WINDOWS" = true ]; then
        python3 speech_to_motion_pipeline.py --select_windows \
            --csv_path $TOK_CSV \
            --output_windows_csv $WINDOWS_CSV \
            --min_window_rms $MIN_WINDOW_RMS
    fi

    python3 vqvae_motion_tokenizer.py \
        --csv_path $TOK_CSV \
        --save_dir $TOK_SAVE_DIR \
        --n_clusters $N_CLUSTERS \
        --epochs $TOK_EPOCHS \
        --batch_size $TOK_BATCH_SIZE \
        --num_quantizers $NUM_QUANTIZERS \
        $TOK_WINDOWS_FLAG \
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
        --output_jsonl $TRAIN_JSONL \
        --min_window_rms $MIN_WINDOW_RMS \
        $TRAIN_WINDOWS_FLAG

    # Held-out validation JSONL, from the same *_test.csv split dataset_to_csv.py already
    # produces when TEST_FRAC > 0 — built here too so STEP 5 can pass it as eval_dataset.
    # Skipped for sample_dataset (TEST_FRAC=0, no split to build from).
    if [ -f "$TRAIN_CSV_TEST" ]; then
        python3 speech_to_motion_pipeline.py --build_dataset \
            --csv_path $TRAIN_CSV_TEST \
            --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pt" \
            --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
            --output_jsonl $VAL_JSONL \
            --min_window_rms $MIN_WINDOW_RMS
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
        --num_quantizers $NUM_QUANTIZERS \
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
        if [ -n "$MAX_SAMPLES" ]; then
            # MAX_SAMPLES set == OVERFIT-SCALE SWEEP run: fixed to BWW760's clip every time
            # (same session/subject the sample_dataset branch below uses) so every overfit
            # run — 4 clips, 100 rows, 1000 rows — is scored against the SAME ground truth.
            # Previously this called select_inference_sample.py to auto-pick the
            # most-dynamic row in whatever got randomly sampled, which meant runs at
            # different scales were being graded on different, non-comparable clips (the
            # 100-row run's pick was 2x more dynamic by that same score than BWW760 — see
            # run-overfit100-q4-sep20 vs run-overfit4-adamw-sep19). --force_session above
            # guarantees BWW760 is actually IN the training set, so this stays a real
            # memorization check, not accidental generalization.
            SESSION_NAME="c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600"
            GT_SUBJECT="BWW760"
            GT_SESSION_DIR="${DATASET_ROOT}/acting/${SESSION_NAME}"
            INFERENCE_AUDIO="${DATASET_ROOT}/acting/${SESSION_NAME}/${GT_SUBJECT}/audio_separated/${SESSION_NAME}.wav"
        elif [ -f "$TRAIN_CSV_TEST" ]; then
            # No cap (the real full run): held-out generalization test, most-dynamic row in
            # the held-out pool so there's something to see in the comparison video.
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

    # If STEP 4 windowed this clip to something other than [0, max_duration_sec) (see
    # select_best_audio_window / MIN_WINDOW_RMS above), inference and eval must use that
    # exact same window or they'd condition on / score against the wrong 10s. 0.0 (i.e. no
    # windowing happened, or the sidecar doesn't exist yet) reproduces the old behavior.
    AUDIO_WINDOWS_CSV="${TRAIN_JSONL%.jsonl}_audio_windows.csv"
    AUDIO_START_SEC="0.0"
    if [ -f "$AUDIO_WINDOWS_CSV" ] && [ -n "$INFERENCE_AUDIO" ]; then
        # os.path.normpath, not a straight string match: dataset_to_csv.py writes paths
        # via pathlib (str(Path("./full_dataset/...")) drops the leading "./"), while
        # INFERENCE_AUDIO here is built by plain bash concatenation (keeps it) -- an exact
        # string compare silently never matches and always falls back to 0.0.
        AUDIO_START_SEC=$(python3 -c "
import csv, os
target = os.path.normpath('$INFERENCE_AUDIO')
with open('$AUDIO_WINDOWS_CSV') as f:
    for row in csv.DictReader(f):
        if os.path.normpath(row['audio_filename']) == target:
            print(row['start_sec']); break
    else:
        print('0.0')
")
    fi

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
        --num_quantizers $NUM_QUANTIZERS \
        --start_sec $AUDIO_START_SEC \
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
        --azim        -60 \
        --gt_start_sec $AUDIO_START_SEC

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
        --gt_start_sec $AUDIO_START_SEC \
        --output_json "$METRICS_JSON"

    echo -e "\n${GREEN}=> Metrics saved: $METRICS_JSON${NC}"
fi
