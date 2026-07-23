#!/bin/bash
# Task #72 Phase 6: 3 general baselines on GPU 2 (idle, BERT4Rec done)
# MF -> Pop -> Random, mode: full ranking
# Uses musical_instruments_general.yaml (general config, NOT sequential)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 2] SEQUENTIAL general-baselines launcher started at $(date) ====="
echo "===== Config: $CONFIG, mode: full, PHYSICAL gpu_id=2 ====="

for MODEL in DMF Pop Random; do
    LOG_FILE="$LOG_DIR/task72_phase6_full_${MODEL}_gpu2.log"
    echo "===== [GPU 2] Starting $MODEL (full) at $(date) =====" > "$LOG_FILE"
    python3 run_recbole_baseline.py --model="$MODEL" --gpu_id=2 --config="$CONFIG" \
        >> "$LOG_FILE" 2>&1
    echo "===== [GPU 2] $MODEL (full) completed at $(date) =====" >> "$LOG_FILE"
    echo "===== [GPU 2] $MODEL done at $(date) ====="
done

echo "===== [GPU 2] All 3 general baselines (MF/Pop/Random) completed at $(date) ====="