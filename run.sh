#!/bin/bash

# Exit immediately if a command crashes
set -e

# ==========================================
# COLOR DEFINITIONS
# ==========================================
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
NC='\033[0m' # No Color (Resets the terminal color)

# ==========================================
# 1) TOKENIZER PHASE CONFIGURATION
# ==========================================
TOK_ROOT_FOLDER="./datasets/tokenizer_dataset"
TOK_CSV="datasets/tokenizer_dataset_mapping.csv"
TOK_SAVE_DIR="motion_tokenizer_artifacts"
N_CLUSTERS=1024

# ==========================================
# 2) FINE-TUNING PHASE CONFIGURATION
# ==========================================
TRAIN_ROOT_FOLDER="./datasets/training_dataset"
TRAIN_CSV="datasets/training_dataset_mapping.csv"
TRAIN_JSONL="datasets/speech_motion_train.jsonl"
OUTPUT_DIR="speech_motion_outputs"
BASE_MODEL="unsloth/orpheus-3b-0.1-pretrained"

# Hyperparameters
MAX_STEPS=2000
LOGGING_STEPS=5
SAVE_STEPS=1000

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}🚀 Starting Speech-to-Motion Pipeline${NC}"
echo -e "${CYAN}==========================================${NC}"

# ---------------------------------------------------------
# STEP 1: Process the Tokenizer Dataset
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [1/5] Generating CSV for Tokenizer Dataset...${NC}"
python3 dataset_to_csv.py \
    --root_folder $TOK_ROOT_FOLDER \
    --output_csv $TOK_CSV

echo -e "\n${YELLOW}=> [2/5] Training K-Means Motion Tokenizer...${NC}"
python3 k_means_motion_tokenizer.py \
    --csv_path $TOK_CSV \
    --save_dir $TOK_SAVE_DIR \
    --n_clusters $N_CLUSTERS \
    --tokenize_jsonl

# ---------------------------------------------------------
# STEP 2: Process the Fine-Tuning Dataset
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [3/5] Generating CSV for Training Dataset...${NC}"
python3 dataset_to_csv.py \
    --root_folder $TRAIN_ROOT_FOLDER \
    --output_csv $TRAIN_CSV

echo -e "\n${YELLOW}=> [4/5] Building Joint Audio-Motion JSONL for LLM...${NC}"
python3 speech_to_motion_pipeline.py --build_dataset \
    --csv_path $TRAIN_CSV \
    --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
    --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
    --output_jsonl $TRAIN_JSONL

# ---------------------------------------------------------
# STEP 3: Train the LLM
# ---------------------------------------------------------
echo -e "\n${YELLOW}=> [5/5] Starting Unsloth LoRA Fine-Tuning...${NC}"
python3 speech_to_motion_pipeline.py --train \
    --output_jsonl $TRAIN_JSONL \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS

echo -e "\n${GREEN}==========================================${NC}"
echo -e "${GREEN}✅ Pipeline Complete! Model saved to $OUTPUT_DIR${NC}"
echo -e "${GREEN}==========================================${NC}"