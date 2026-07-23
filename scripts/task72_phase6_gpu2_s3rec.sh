#!/bin/bash
# Task #72 Phase 6u: S3Rec sequential baseline on GPU 2 (idle after FOSSIL done)
# S3Rec = Sequential self-Supervised Rec (Mingdeng et al. 2022, SIGIR)
# Uses mutual information maximization

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] S3Rec launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_S3Rec_gpu2.log"
echo "===== [GPU 2] Starting S3Rec (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=S3Rec --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] S3Rec (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] S3Rec completed at $(date) ====="
