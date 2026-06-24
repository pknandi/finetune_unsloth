#!/bin/bash

# Exit immediately if a command crashes
set -e

# ==========================================
# EXPORTS
# ==========================================
export CUDA_VISIBLE_DEVICES=1

ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
    echo "Loading env from $ENV_FILE"

    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: Env file not found: $ENV_FILE"
fi

# ==========================================
# STAGE CONTROL
# ==========================================
STAGE=${STAGE:-1}
STOP_STAGE=${STOP_STAGE:-999}

echo "Running with STAGE=$STAGE STOP_STAGE=$STOP_STAGE"

# ==========================================
# RUN + LOGGING
# ==========================================
RUN_NAME="run-may13"

LOG_DIR="./outputs/terminal"
LOG_FILE="$LOG_DIR/$(date '+%Y-%m-%d_%H-%M-%S').log"

mkdir -p "./outputs/$RUN_NAME"
mkdir -p "$LOG_DIR"
mkdir -p "inference_data/output"

# Save stdout + stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging terminal output to: $LOG_FILE"

# ==========================================
# COLORS
# ==========================================
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No color

# ==========================================
# TOKENIZER CONFIG
# ==========================================

TOK_ROOT_FOLDER="./Motion_speech_project/tokenizer_data"              # Tokenizer training dataset
TOK_CSV="./outputs/$RUN_NAME/tokenizer_dataset_mapping.csv"          # audio_filename,motion_dirname
TOK_SAVE_DIR="./outputs/$RUN_NAME/motion_tokenizer_artifacts"        # tokenizer.pkl + normalizer.npz

N_CLUSTERS=1024                                                       # Motion token vocabulary size

# ==========================================
# TRAINING CONFIG
# ==========================================

TRAIN_ROOT_FOLDER="./Motion_speech_project/tokenizer_data"           # LLM training dataset
TRAIN_CSV="./outputs/$RUN_NAME/training_dataset_mapping.csv"         # Training dataset CSV
TRAIN_JSONL="./outputs/$RUN_NAME/speech_motion_train.jsonl"          # Final tokenized dataset

OUTPUT_DIR="./outputs/$RUN_NAME/lora"                                # LoRA checkpoints
BASE_MODEL="unsloth/llama-3-8b-bnb-4bit"                             # Base model
RESUME_CHECKPOINT="" #"./outputs/$RUN_NAME/lora/checkpoint-2500"

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================

MAX_STEPS=10000
LOGGING_STEPS=5
SAVE_STEPS=2500

# ==========================================
# INFERENCE CONFIG
# ==========================================

AUDIO_FILE_NAME="c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.wav"  # Just the filename, not full path
AUDIO_BASENAME=$(basename "$AUDIO_FILE_NAME" .wav)                   # abc.wav -> abc
INFERENCE_AUDIO="/home/mubtasim/speech-motion/Motion_speech_project/motion_speech_dataset/audio/c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600/BWW760/audio_separated/c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.wav"
INFERECE_CHECKPOINT="./outputs/$RUN_NAME/lora/checkpoint-2500"
INFERENCE_OUTPUT="./outputs/$RUN_NAME/inference/generated_motion_${RUN_NAME}_$(echo "$BASE_MODEL" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_\|_$//g')_${AUDIO_BASENAME}.npy"
INFERENCE_VIDEO="./outputs/$RUN_NAME/inference/video_${RUN_NAME}_$(echo "$BASE_MODEL" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_\|_$//g')_${AUDIO_BASENAME}.mp4"
SMPLX_MODEL_DIR="./outputs/smplx/models"

# Ground truth: session dir (contains BWW760, DXG448, etc. sub-dirs) + subject
GT_SESSION_DIR="./Motion_speech_project/motion_speech_dataset/smplx/c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600"
GT_SUBJECT="BWW760"
COMPARISON_VIDEO="./outputs/$RUN_NAME/inference/comparison_${RUN_NAME}_${BASE_MODEL_SLUG}_${AUDIO_BASENAME}.mp4"


#################################################################################################################
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 Starting Full Speech-to-Motion Pipeline${NC}"
echo -e "${CYAN}==========================================${NC}"

# ---------------------------------------------------------
# STEP 1: Process the Tokenizer Dataset
# ---------------------------------------------------------
if [ "$STAGE" -le 1 ] && [ "$STOP_STAGE" -ge 1 ]; then
    echo -e "\n${YELLOW}=> [1/6] Generating CSV for Tokenizer Dataset...${NC}"

    python3 dataset_to_csv.py \
        --root_folder $TOK_ROOT_FOLDER \
        --output_csv $TOK_CSV
fi

# ---------------------------------------------------------
# STEP 2: Train K-Means Motion Tokenizer
# ---------------------------------------------------------
if [ "$STAGE" -le 2 ] && [ "$STOP_STAGE" -ge 2 ]; then
    echo -e "\n${YELLOW}=> [2/6] Training K-Means Motion Tokenizer...${NC}"

    python3 k_means_motion_tokenizer.py \
        --csv_path $TOK_CSV \
        --save_dir $TOK_SAVE_DIR \
        --n_clusters $N_CLUSTERS \
        --tokenize_jsonl
fi

# ---------------------------------------------------------
# STEP 3: Process the Fine-Tuning Dataset
# ---------------------------------------------------------
if [ "$STAGE" -le 3 ] && [ "$STOP_STAGE" -ge 3 ]; then
    echo -e "\n${YELLOW}=> [3/6] Generating CSV for Training Dataset...${NC}"

    python3 dataset_to_csv.py \
        --root_folder $TRAIN_ROOT_FOLDER \
        --output_csv $TRAIN_CSV
fi

# ---------------------------------------------------------
# STEP 4: Build Joint Audio-Motion JSONL
# ---------------------------------------------------------
if [ "$STAGE" -le 4 ] && [ "$STOP_STAGE" -ge 4 ]; then
    echo -e "\n${YELLOW}=> [4/6] Building Joint Audio-Motion JSONL for LLM...${NC}"

    python3 speech_to_motion_pipeline.py --build_dataset \
        --csv_path $TRAIN_CSV \
        --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
        --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
        --output_jsonl $TRAIN_JSONL
fi

# ---------------------------------------------------------
# STEP 5: Train the LLM
# ---------------------------------------------------------
if [ "$STAGE" -le 5 ] && [ "$STOP_STAGE" -ge 5 ]; then
    echo -e "\n${YELLOW}=> [5/6] Starting Unsloth LoRA Fine-Tuning...${NC}"

    python3 speech_to_motion_pipeline.py --train \
        --output_jsonl $TRAIN_JSONL \
        --output_dir $OUTPUT_DIR \
        --base_model $BASE_MODEL \
        --max_steps $MAX_STEPS \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        --resume_from_checkpoint $RESUME_CHECKPOINT
fi

# ---------------------------------------------------------
# STEP 6: Inference
# ---------------------------------------------------------
if [ "$STAGE" -le 6 ] && [ "$STOP_STAGE" -ge 6 ]; then
    echo -e "\n${MAGENTA}=> [6/6] Running Inference on Test Audio...${NC}"

    # python3 run_inference.py 
    python3 speech_to_motion_inference.py\
        --audio_path $INFERENCE_AUDIO \
        --lora_model_dir "$INFERECE_CHECKPOINT" \
        --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
        --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
        --base_model $BASE_MODEL \
        --output_npy_path $INFERENCE_OUTPUT


    echo -e "\n${GREEN}==========================================${NC}"
    echo -e "${GREEN}✅ Pipeline Complete!${NC}"
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}Motion output saved to: $INFERENCE_OUTPUT${NC}"
    echo -e "${GREEN}==========================================${NC}"

fi

# ---------------------------------------------------------
# STEP 7: Visualization (predicted motion only)
# ---------------------------------------------------------
if [ "$STAGE" -le 7 ] && [ "$STOP_STAGE" -ge 7 ]; then
    echo -e "\n${CYAN}=> [7/8] Rendering SMPL-X Motion Visualization...${NC}"
 
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
    echo -e "\n${CYAN}=> [8/8] Rendering GT vs Predicted Comparison Video...${NC}"

    # Diagnostics first — prints frame counts, no rendering
    echo -e "${YELLOW}  -- Diagnostics --${NC}"
    python3 compare_motion.py --diagnose \
        --session_dir "$GT_SESSION_DIR" \
        --subject     "$GT_SUBJECT" \
        --pred_npy    "$INFERENCE_OUTPUT"

    # Render side-by-side
    python3 compare_motion.py \
        --session_dir "$GT_SESSION_DIR" \
        --subject     "$GT_SUBJECT" \
        --pred_npy    "$INFERENCE_OUTPUT" \
        --audio_path  "$INFERENCE_AUDIO" \
        --output_path "$COMPARISON_VIDEO" \
        --fps         30 \
        --max_seconds 10.0 \
        --width       1280 \
        --height      540 \
        --elev        15 \
        --azim        -60

    echo -e "\n${GREEN}=> Comparison video: $COMPARISON_VIDEO${NC}"
fi