#!/bin/bash
# Task #81 — S³Rec RecBole 训练 (paper Table 2 #8, paper R@10=0.0538)
set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env

cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

GPU=${CUDA_VISIBLE_DEVICES:-0}
LOG_FILE=/home/wlia0047/ar57/wenyu/GeneRec/logs/task81_s3rec_jul-23-2026_13-25-00.log

echo "===== Starting S3Rec on GPU $GPU at $(date) =====" | tee -a $LOG_FILE
python3 run_recbole_baseline.py --model=S3Rec --gpu_id=$GPU \
    --config=musical_instruments_sequential_paper.yaml \
    >> $LOG_FILE 2>&1
echo "===== S3Rec completed at $(date) =====" | tee -a $LOG_FILE
