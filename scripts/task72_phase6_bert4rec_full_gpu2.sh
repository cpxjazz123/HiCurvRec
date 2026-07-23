#!/bin/bash
# Task #72 Phase 6: BERT4Rec full ranking on GPU 2 (single model launcher)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] BERT4Rec (full) launcher started at $(date) ====="
LOG_FILE="$LOG_DIR/task72_phase6_full_BERT4Rec_gpu2.log"
echo "===== [GPU 2] Starting BERT4Rec (full) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=BERT4Rec --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 2] BERT4Rec (full) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] BERT4Rec (full) done at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] BERT4Rec done at $(date) ====="