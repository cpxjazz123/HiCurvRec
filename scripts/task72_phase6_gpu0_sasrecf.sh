#!/bin/bash
# Task #72 Phase 6ab: SASRecF sequential baseline on GPU 0 (idle after HRM done)
# SASRecF = SASRec Frequency-aware variant

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 0] SASRecF launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_SASRecF_gpu0.log"
echo "===== [GPU 0] Starting SASRecF (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=SASRecF --gpu_id=0 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 0] SASRecF (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 0] SASRecF completed at $(date) ====="