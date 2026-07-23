#!/bin/bash
# Task #72 Phase 6aa: GRU4RecCPR sequential baseline on GPU 0 (idle after FEARec done)
# GRU4RecCPR = GRU4Rec Context-aware Personalized Recommendation

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 0] GRU4RecCPR launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_GRU4RecCPR_gpu0.log"
echo "===== [GPU 0] Starting GRU4RecCPR (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=GRU4RecCPR --gpu_id=0 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 0] GRU4RecCPR (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 0] GRU4RecCPR completed at $(date) ====="