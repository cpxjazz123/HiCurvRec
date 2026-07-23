#!/bin/bash
# Task #72 paper-aligned rerun: Caser on GPU 0 (idle)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_sequential_paper.yaml"

echo "===== [GPU 0] Caser paper-aligned started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_paper_aligned_Caser_gpu0.log"
echo "===== [GPU 0] Caser paper-aligned launched at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model="Caser" --gpu_id=0 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 0] Caser paper-aligned completed at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 0] Caser done at $(date) ====="
