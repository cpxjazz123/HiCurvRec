#!/bin/bash
# Task #184 Stage 2 — Product manifold codebook inference (Sinkhorn-Knopp + dedup)
# 启动 GPU 1 (Task #185 训练占用 GPU 0, GPU 1/2/3 空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task184

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task184
mkdir -p /fs04/ar57/wenyu/GeneRec/logs/task184

echo "===== [Task #184 Stage 2] Product manifold codebook inference at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task184_stage2_codebook.py 2>&1 | tee /fs04/ar57/wenyu/GeneRec/logs/task184/stage2_codebook.out

echo "===== [Task #184 Stage 2] completed at $(date) ====="
