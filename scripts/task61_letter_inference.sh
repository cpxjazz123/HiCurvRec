#!/bin/bash
# Task #61 — LETTER-TIGER inference on Musical_Instruments test set
# Uses checkpoint-15222 (best_metric=1.588, eval_loss=1.588)
# Constrained beam search (num_beams=20) with Trie prefix_allowed_tokens

set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task61_letter_inference_${TS}.log"

RESULTS_DIR="/home/wlia0047/ar57/wenyu/GeneRec/results/task61_letter_inference"
mkdir -p "$RESULTS_DIR"

mkdir -p "$LOG_DIR"

echo "===== [Task #61] LETTER-TIGER inference launched at $(date) =====" | tee "$LOG_FILE"
echo "Using checkpoint-15222 (best_metric=1.588, eval_loss=1.588)" | tee -a "$LOG_FILE"

# GPU 1 is free (GPU 0 is busy with ETEGRec 896d)
export CUDA_VISIBLE_DEVICES=1

python3 test.py \
    --gpu_id 0 \
    --ckpt_path ./ckpt/Instruments/checkpoint-15222 \
    --dataset Instruments \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data \
    --results_file ${RESULTS_DIR}/letter_tiger_instruments_test.json \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.epoch5000.alpha0.01-beta0.0001.json \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #61] LETTER-TIGER inference completed at $(date) =====" | tee -a "$LOG_FILE"