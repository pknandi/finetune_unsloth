#!/bin/bash

# Exit immediately if a command crashes
set -e

# ==========================================
# 1) TOKENIZER PHASE CONFIGURATION
# ==========================================
TOK_ROOT_FOLDER="./datasets/tokenizer_dataset_may13"
TOK_CSV="datasets/tokenizer_dataset_mapping_may13.csv"
TOK_SAVE_DIR="motion_tokenizer_artifacts_may13"
N_CLUSTERS=1024

# ==========================================
# 2) FINE-TUNING PHASE CONFIGURATION
# ==========================================
TRAIN_ROOT_FOLDER="./datasets/training_dataset_may13"
TRAIN_CSV="datasets/training_dataset_mapping_may13.csv"
TRAIN_JSONL="datasets/speech_motion_train_may13.jsonl"
OUTPUT_DIR="speech_motion_outputs_may13"
BASE_MODEL="unsloth/orpheus-3b-0.1-pretrained"

# Hyperparameters
MAX_STEPS=10
LOGGING_STEPS=5
SAVE_STEPS=10


echo "=========================================="
echo " Starting Speech-to-Motion Pipeline"
echo "=========================================="

# ---------------------------------------------------------
# STEP 1: Process the Tokenizer Dataset
# ---------------------------------------------------------
echo "=> [1/5] Generating CSV for Tokenizer Dataset..."
python3 dataset_to_csv.py \
    --root_folder $TOK_ROOT_FOLDER \
    --output_csv $TOK_CSV

echo "=> [2/5] Training K-Means Motion Tokenizer..."
python3 k_means_motion_tokenizer.py \
    --csv_path $TOK_CSV \
    --save_dir $TOK_SAVE_DIR \
    --n_clusters $N_CLUSTERS \
    --tokenize_jsonl  # Include this flag to generate the debug JSONL

# ---------------------------------------------------------
# STEP 2: Process the Fine-Tuning Dataset
# ---------------------------------------------------------
echo "=> [3/5] Generating CSV for Training Dataset..."
python3 dataset_to_csv.py \
    --root_folder $TRAIN_ROOT_FOLDER \
    --output_csv $TRAIN_CSV

echo "=> [4/5] Building Joint Audio-Motion JSONL for LLM..."
python3 speech_to_motion_pipeline.py --build_dataset \
    --csv_path $TRAIN_CSV \
    --tokenizer_path "$TOK_SAVE_DIR/tokenizer.pkl" \
    --normalizer_path "$TOK_SAVE_DIR/normalizer.npz" \
    --output_jsonl $TRAIN_JSONL

# ---------------------------------------------------------
# STEP 3: Train the LLM
# ---------------------------------------------------------
echo "=> [5/5] Starting Unsloth LoRA Fine-Tuning..."
python3 speech_to_motion_pipeline.py --train \
    --output_jsonl $TRAIN_JSONL \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --max_steps $MAX_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS

echo "=========================================="
echo " Pipeline Complete! Model saved to $OUTPUT_DIR"
echo "=========================================="