#!/bin/bash
# One tokenizer-only training job on the 1000-row windows (no LLM). Resumes from
# <save_dir>/train_checkpoint.pt, so a larger <epochs> continues an earlier job.
# Writes the exit code to <save_dir>.exit when it finishes (scripts/orchestrate_1000.sh waits on it).
# usage: CUDA_VISIBLE_DEVICES=<gpu> scripts/train_tokenizer_job.sh <save_dir> <num_quantizers> <hidden_dim> <epochs> [acc_weight=0]
SAVE_DIR=$1; Q=$2; HIDDEN=$3; EPOCHS=$4; ACC=${5:-0}
BASE=${BASE:-outputs/run-overfit1000-tokfix-oct17}     # stage-1 CSV + audio_windows.csv (default: the 1000-row set)
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
# The model and loss are on the GPU, but torch's default CPU thread pool spun ~38 cores per job
# (GPU at 3 %) on this shared 96-core box; two jobs in parallel ran 3x slower. Cap it.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
rm -f "$SAVE_DIR.exit"
python3 vqvae_motion_tokenizer.py \
    --csv_path $BASE/tokenizer_dataset_mapping.csv \
    --save_dir "$SAVE_DIR" \
    --n_clusters 1024 \
    --epochs "$EPOCHS" \
    --batch_size 64 \
    --num_quantizers "$Q" \
    --hidden_dim "$HIDDEN" \
    --acc_weight "$ACC" \
    --windows_csv $BASE/audio_windows.csv 2>&1 | tee -a "$SAVE_DIR.log"
echo "${PIPESTATUS[0]}" > "$SAVE_DIR.exit"
