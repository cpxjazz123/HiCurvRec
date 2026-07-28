#!/bin/bash
# Task #140 — S³Rec Stage 1 pretrain launcher (background daemon)
# 2026-07-24
#
# R7: GPU 0 free (0% util, 0 MiB).
# R12: RecBole S3Rec built-in checkpoint saving every save_step; pretrain ckpts auto-named.
# Background execution to free this loop tick.
set -uo pipefail

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task140
mkdir -p "$LOG_DIR"
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task140/_STAGE1_PID
mkdir -p "$(dirname "$PID_FILE")"
PRODUCTS_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/pretrain
mkdir -p "$PRODUCTS_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/s3rec_pretrain_${TS}.log"

echo "[$(date)] Launching Task #140 Stage 1 (pretrain 50 epoch) — log: $LOG_FILE"

# run via nohup so it survives this loop tick exit
nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task140_s3rec_stage1_pretrain.py \
    --epochs 50 --save_step 10 --seed 2025 --gpu_id 0 \
    > "$LOG_FILE" 2>&1 &

S3REC_PID=$!
echo "$S3REC_PID" > "$PID_FILE"
echo "[$(date)] PID=$S3REC_PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
