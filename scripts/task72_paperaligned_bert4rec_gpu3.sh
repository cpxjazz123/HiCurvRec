#!/bin/bash
# Task #72 paper-aligned rerun: BERT4Rec on GPU 3 (idle)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_sequential_paper.yaml"

echo "===== [GPU 3] BERT4Rec paper-aligned started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_paper_aligned_BERT4Rec_gpu3.log"
echo "===== [GPU 3] BERT4Rec paper-aligned launched at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model="BERT4Rec" --gpu_id=3 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 3] BERT4Rec paper-aligned completed at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 3] BERT4Rec done at $(date) ====="
