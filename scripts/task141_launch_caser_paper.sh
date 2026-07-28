#!/bin/bash
# Task #141 — Caser paper-aligned fix launcher (background)
# 2026-07-24
#
# R7: GPU 3 free (0% util).
# R12: RecBole built-in best valid ckpt saving.
set -uo pipefail

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task141
mkdir -p "$LOG_DIR"
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task141/_TRAINING_PID
PRODUCTS_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task141/train
mkdir -p "$PRODUCTS_DIR" "$(dirname "$PID_FILE")"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/caser_paper_aligned_${TS}.log"

echo "[$(date)] Launching Task #141 Caser paper-aligned fix (lr=0.001, wd=0.0) — log: $LOG_FILE"

nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task141_caser_paper_aligned_train.py \
    > "$LOG_FILE" 2>&1 &

CASER_PID=$!
echo "$CASER_PID" > "$PID_FILE"
echo "[$(date)] PID=$CASER_PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
