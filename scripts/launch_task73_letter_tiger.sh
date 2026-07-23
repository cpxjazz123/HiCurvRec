#!/bin/bash
# Task #73 — LETTER pipeline: tokenize + TIGER training (Stage 2)
# Prereq: LETTER RQ-VAE training (launch_task73_letter.sh) has produced ckpt
set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Stage 2.1: Generate SID indices ===
LOG_FILE="$LOG_DIR/task73_letter_indices.log"
echo "===== [Task #73] LETTER generate_indices started at $(date) =====" | tee "$LOG_FILE"

# Find the latest RQ-VAE checkpoint (NOTE: ckpt saved to ../checkpoint due to RQ-VAE cwd, NOT inside LETTER/)
CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/external/checkpoint/Instruments_bge_sasrec32"
LATEST_RUN=$(ls -dt "$CKPT_DIR"/*/ 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: no RQ-VAE checkpoint run found in $CKPT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
LATEST_CKPT=$(ls -t "$LATEST_RUN"/epoch_*_collision_*.pth 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(ls "$LATEST_RUN"/best_loss_model.pth "$LATEST_RUN"/best_collision_model.pth 2>/dev/null | head -1)
fi
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: no RQ-VAE checkpoint found in $CKPT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Using checkpoint: $LATEST_CKPT" | tee -a "$LOG_FILE"

# Convert to absolute path
CKPT_ABS=$(realpath "$LATEST_CKPT")

# generate_indices.py outputs .index.json in LETTER/LETTER-TIGER/ckpt/Instruments/
python -u RQ-VAE/generate_indices.py \
    --dataset Instruments \
    --alpha 0.01 \
    --beta 0.0001 \
    --epoch 5000 \
    --checkpoint "$CKPT_ABS" \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] LETTER generate_indices completed at $(date) =====" | tee -a "$LOG_FILE"

# === Stage 2.2: LETTER-TIGER training ===
LOG_FILE="$LOG_DIR/task73_letter_tiger.log"
echo "===== [Task #73] LETTER-TIGER started at $(date) =====" | tee "$LOG_FILE"

cd LETTER-TIGER

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET/
mkdir -p "$OUTPUT_DIR"

# Find the .index.json file (in LETTER/data/Instruments/, NOT in LETTER-TIGER/)
INDEX_FILE=$(ls ../data/$DATASET/$DATASET.index.*.json 2>/dev/null | head -1)
if [ -z "$INDEX_FILE" ]; then
    echo "ERROR: no .index.json found in ../data/$DATASET/" | tee -a "$LOG_FILE"
    echo "  search: ../data/$DATASET/$DATASET.index.*.json" | tee -a "$LOG_FILE"
    exit 1
fi
INDEX_FILE=$(realpath "$INDEX_FILE")
echo "Using index: $INDEX_FILE" | tee -a "$LOG_FILE"

# Use torchrun 1 GPU (GPU 1 only, GPU 0/2/3 occupied)
torchrun --nproc_per_node=1 --master_port=2314 ./finetune.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file "$INDEX_FILE" \
    --temperature 1.0 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] LETTER-TIGER completed at $(date) =====" | tee -a "$LOG_FILE"