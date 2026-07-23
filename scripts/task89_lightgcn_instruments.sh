#!/bin/bash
# Task #89 — LightGCN RecBole 训练 (paper Table 2 #5, paper R@10=0.0454)
# GPU 推荐: 释放后 GPU (task80/#82/#83 完成后)
# 单 GPU ~30 min (24k items, 511k interactions)
set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env

cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

GPU=${CUDA_VISIBLE_DEVICES:-0}
LOG_FILE=/home/wlia0047/ar57/wenyu/GeneRec/logs/task89_lightgcn_jul-23-2026_13-45-00.log

echo "===== Starting LightGCN (paper Table 2 #5) on GPU $GPU at $(date) =====" | tee -a $LOG_FILE
python3 run_recbole_baseline.py --model=LightGCN --gpu_id=$GPU \
    --config=musical_instruments_general.yaml \
    >> $LOG_FILE 2>&1
echo "===== LightGCN completed at $(date) =====" | tee -a $LOG_FILE
