#!/bin/bash
# Task #139 — LETTER-TIGER paper-aligned baseline launcher
# 2026-07-24
#
# R11.4 dry-run already done with user approval (选项 A).
# Per R7 GPU usage: GPU 0 (free, 0% util), 不抢 Task #137 GPU 1 / Task #136 GPU 2.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task139
mkdir -p "$LOG_DIR"

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER

TRAIN_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task139/_TRAINING_PID
LOG_FILE="$LOG_DIR/letter_tiger_paper_baseline_$(date +%Y-%m-%d_%H-%M-%S).log"

# Activate grid_toys env (torch 2.6+cu124, transformers 4.57.3 — already verified)
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
echo "[$(date)] grid_toys env: $(which python3) | torch=$(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"

# Single GPU 0
bash run_train.sh 2>&1 | tee -a "$LOG_FILE"
TRAIN_PID=$!
echo "$TRAIN_PID" > "$TRAIN_PID_FILE"
echo "[$(date)] training started, PID=$TRAIN_PID, log=$LOG_FILE"
