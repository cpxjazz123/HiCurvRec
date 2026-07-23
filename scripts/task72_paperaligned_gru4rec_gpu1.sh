#!/bin/bash
# Task #72 paper-aligned rerun: GRU4Rec on GPU 1 (idle)
# Uses paper-aligned sequential config (lr=0.003, wd=0.05, max_seq=20, NDCG@10, patience=20)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_sequential_paper.yaml"

echo "===== [GPU 1] GRU4Rec (paper config) started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_paper_aligned_GRU4Rec_gpu1.log"
echo "===== [GPU 1] GRU4Rec paper-aligned launched at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model="GRU4Rec" --gpu_id=1 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 1] GRU4Rec paper-aligned completed at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 1] GRU4Rec done at $(date) ====="
