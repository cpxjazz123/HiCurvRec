#!/bin/bash
# Task #72 paper-aligned rerun: SASRec on GPU 2 (idle)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_sequential_paper.yaml"

echo "===== [GPU 2] SASRec paper-aligned started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_paper_aligned_SASRec_gpu2.log"
echo "===== [GPU 2] SASRec paper-aligned launched at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model="SASRec" --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] SASRec paper-aligned completed at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] SASRec done at $(date) ====="
