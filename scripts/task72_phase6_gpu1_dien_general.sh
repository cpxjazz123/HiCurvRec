#!/bin/bash
# Task #72 Phase 6af: DIEN general yaml retry on GPU 1
# DIEN = Deep Interest Evolution Network (Zhou et al. 2019, AAAI)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 1] DIEN general yaml launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_DIEN_general_gpu1.log"
echo "===== [GPU 1] Starting DIEN (general yaml) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=DIEN --gpu_id=1 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 1] DIEN (general) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 1] DIEN completed at $(date) ====="