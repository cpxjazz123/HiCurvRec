#!/bin/bash
# Task #72 Phase 6: 3 general baselines on GPU 0 (idle, DECOR done)
# BPR -> NeuMF -> LightGCN, mode: full ranking
# Uses musical_instruments_general.yaml (general config, NOT sequential)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 0] SEQUENTIAL general-baselines launcher started at $(date) ====="
echo "===== Config: $CONFIG, mode: full, PHYSICAL gpu_id=0 ====="

for MODEL in BPR NeuMF LightGCN; do
    LOG_FILE="$LOG_DIR/task72_phase6_full_${MODEL}_gpu0.log"
    echo "===== [GPU 0] Starting $MODEL (full) at $(date) =====" > "$LOG_FILE"
    python3 run_recbole_baseline.py --model="$MODEL" --gpu_id=0 --config="$CONFIG" \
        >> "$LOG_FILE" 2>&1
    echo "===== [GPU 0] $MODEL (full) completed at $(date) =====" >> "$LOG_FILE"
    echo "===== [GPU 0] $MODEL done at $(date) ====="
done

echo "===== [GPU 0] All 3 general baselines (BPR/NeuMF/LightGCN) completed at $(date) ====="