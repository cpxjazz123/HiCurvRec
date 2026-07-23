#!/bin/bash
# Task #72 Phase 6: Controlled BERT4Rec full ranking on GPU 2 (idle)
# Replaces rogue task72_phase6_bert4rec_full_gpu2.sh spawn
# BERT4Rec uses --gpu_id=2 (physical) because RecBole overwrites CUDA_VISIBLE_DEVICES
# Runs in parallel with Caser (GPU 3) — does NOT interfere

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
LOG_FILE="$LOG_DIR/task72_phase6_full_BERT4Rec_gpu2.log"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2 CONTROLLED] BERT4Rec full ranking launched at $(date) =====" > "$LOG_FILE"
echo "===== Config: $CONFIG, mode: full, PHYSICAL gpu_id=2 =====" >> "$LOG_FILE"

python3 run_recbole_baseline.py --model=BERT4Rec --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1

echo "===== [GPU 2] BERT4Rec full ranking completed at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] BERT4Rec done at $(date) ====="