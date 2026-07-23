#!/bin/bash
# Task #72 Phase 6v: GCSAN sequential baseline on GPU 2 (idle after FOSSIL done)
# GCSAN = Graph Contextualized Self-Attention Network (Xu et al. 2020)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] GCSAN launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_GCSAN_gpu2.log"
echo "===== [GPU 2] Starting GCSAN (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=GCSAN --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] GCSAN (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] GCSAN completed at $(date) ====="
