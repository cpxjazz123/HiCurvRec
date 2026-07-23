#!/bin/bash
# Task #73 SASRec 32-d retrain launcher (genrec_env)
# 用法: bash launch_task73_sasrec32_retrain.sh <gpu_id>
set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

mkdir -p logs

GPU="${1:-1}"
echo "===== [Task #73] SASRec 32-d retrain launched at $(date) =====" | tee -a logs/task73_sasrec32_retrain.log
echo "GPU: $GPU" | tee -a logs/task73_sasrec32_retrain.log

# 使用 genrec_env 的 python 直接调用,避免 PATH 失效
$CONDA_PREFIX/bin/python3 scripts/task73_sasrec32_retrain.py --gpu_id "$GPU" 2>&1 | tee -a logs/task73_sasrec32_retrain.log
echo "===== [Task #73] SASRec 32-d retrain done at $(date) =====" | tee -a logs/task73_sasrec32_retrain.log