#!/bin/bash

# Exit immediately if a command crashes
set -e

# ==========================================
# COLOR DEFINITIONS
# ==========================================
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No Color

# ==========================================
# 1) TOKENIZER PHASE CONFIGURATION
# ==========================================
TOK_ROOT_FOLDER="./datasets/tokenizer_dataset_may13"
TOK_CSV="datasets/tokenizer_dataset_mapping_may13.csv"
TOK_SAVE_DIR="motion_tokenizer_artifacts"
N_CLUSTERS=1024

# ==========================================
# 2) FINE-TUNING PHASE CONFIGURATION
# ==========================================
TRAIN_ROOT_FOLDER="./datasets/training_dataset_may13"
TRAIN_CSV="datasets/training_dataset_mapping_may13.csv"
TRAIN_JSONL="datasets/speech_motion_train_may13.jsonl"
OUTPUT_DIR="speech_motion_outputs_may13"
BASE_MODEL="unsloth/orpheus-3b-0.1-pretrained"

MAX_STEPS=100
LOGGING_STEPS=5
SAVE_STEPS=50

# ==========================================
# 3) INFERENCE PHASE CONFIGURATION
# ==========================================
AUDIO_FILE_NAME="c--20250122--1350--ZPZ640--HXR046--FGI958--DLF703--pilot--MotionPrior--DAYLIFE_Doing_chores_together--186171-190520.wav"
INFERENCE_AUDIO="inference_data/input/$AUDIO_FILE_NAME"

# Bash trick to extract the filename without the .wav extension
AUDIO_BASENAME=$(basename "$AUDIO_FILE_NAME" .wav)
INFERENCE_OUTPUT="inference_data/output/generated_motion_${AUDIO_BASENAME}.npy"


echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 Starting Full Speech-to-Motion Pipeline${NC}"
echo -e "${CYAN}==========================================${NC}"

# ---------------------------------------------------------
# STEP 1: Process the Tokenizer Dataset
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [1/6] Generating CSV for Tokenizer Dataset...${NC}"
python3 dataset_to_csv.py \
    --root_folder $TOK_ROOT_FOLDER \
    --output_csv $TOK_CSV

echo -e "\n${YELLOW}=> [2/6] Training K-Means Motion Tokenizer...${NC}"
python3 k_means_motion_tokenizer.py \
    --csv_path $TOK_CSV \
    --save_dir $TOK_SAVE_DIR \
    --n_clusters $N_CLUSTERS \
    --tokenize_jsonl

# ---------------------------------------------------------
# STEP 2: Process the Fine-Tuning Dataset
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [3/6] Generating CSV for Training Dataset...${NC}"
python3 dataset_to_csv.py \
    --root_folder $TRAIN_ROOT_FOLDER \
    --output_csv $TRAIN_CSV

echo -e "\n${YELLOW}=> [4/6] Building Joint Audio-Motion JSONL for LLM...${NC}"
python3 speech_to_motion_pipeline.py --build_dataset \
    --csv_path $TRAIN_CSV \
    --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
    --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
    --output_jsonl $TRAIN_JSONL

# ---------------------------------------------------------
# STEP 3: Train the LLM
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [5/6] Starting Unsloth LoRA Fine-Tuning...${NC}"
python3 speech_to_motion_pipeline.py --train \
    --output_jsonl $TRAIN_JSONL \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS

# ---------------------------------------------------------
# STEP 4: Inference
# ---------------------------------------------------------
echo -e "\n${MAGENTA}=> [6/6] Running Inference on Test Audio...${NC}"
python3 run_inference.py \
    --audio_path $INFERENCE_AUDIO \
    --lora_model_dir "$OUTPUT_DIR/lora" \
    --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
    --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
    --output_npy_path $INFERENCE_OUTPUT


echo -e "\n${GREEN}==========================================${NC}"
echo -e "${GREEN}✅ Pipeline Complete!${NC}"
echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
echo -e "${GREEN}Motion output saved to: $INFERENCE_OUTPUT${NC}"
echo -e "${GREEN}==========================================${NC}"