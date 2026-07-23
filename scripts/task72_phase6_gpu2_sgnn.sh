#!/bin/bash
# Task #72 Phase 6q: SRGNN sequential baseline on GPU 2 (idle after LightGCN done)
# SRGNN = Session RNN with Graph Neural Network (Wu et al. 2019)
# Sequential recommender from RecBole/models/sequential_recommender

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] SRGNN launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_SRGNN_gpu2.log"
echo "===== [GPU 2] Starting SRGNN (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=SRGNN --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] SRGNN (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] SRGNN completed at $(date) ====="
