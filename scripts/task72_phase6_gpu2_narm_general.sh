#!/bin/bash
# Task #72 Phase 6ag: NARM general yaml retry on GPU 2 (idle after RepeatNet done)
# NARM general mode

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 2] NARM general yaml launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_NARM_general_gpu2.log"
echo "===== [GPU 2] Starting NARM (general yaml) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=NARM --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] NARM (general) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] NARM completed at $(date) ====="