#!/bin/bash
# Task #72 Phase 6: SEQUENTIAL full-ranking launcher on GPU 3 (GRU4Rec -> Caser -> BERT4Rec)
# User mandate (2026-07-21): mode: full, NO subset. Sequential per GPU to avoid eval OOM.

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

# CRITICAL: RecBole overwrites CUDA_VISIBLE_DEVICES — must use PHYSICAL gpu_id
echo "===== [GPU 3] SEQUENTIAL full-ranking launcher started at $(date) ====="
echo "===== Config: $CONFIG, mode: full, PHYSICAL gpu_id=3 ====="

for MODEL in GRU4Rec Caser BERT4Rec; do
    LOG_FILE="$LOG_DIR/task72_phase6_full_${MODEL}_gpu3.log"
    echo "===== [GPU 3] Starting $MODEL (full) at $(date) =====" > "$LOG_FILE"
    python3 run_recbole_baseline.py --model="$MODEL" --gpu_id=3 --config="$CONFIG" \
        >> "$LOG_FILE" 2>&1
    echo "===== [GPU 3] $MODEL (full) completed at $(date) =====" >> "$LOG_FILE"
    echo "===== [GPU 3] $MODEL done at $(date) ====="
done

echo "===== [GPU 3] All 3 baselines (GRU4Rec/Caser/BERT4Rec) completed at $(date) ====="