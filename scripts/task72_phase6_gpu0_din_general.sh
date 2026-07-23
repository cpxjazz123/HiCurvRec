#!/bin/bash
# Task #72 Phase 6af: DIN general yaml retry on GPU 0
# DIN = Deep Interest Network (Zhou et al. 2018, KDD)
# Needs general yaml (label field for negative sampling)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 0] DIN general yaml launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_DIN_general_gpu0.log"
echo "===== [GPU 0] Starting DIN (general yaml) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=DIN --gpu_id=0 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 0] DIN (general) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 0] DIN completed at $(date) ====="