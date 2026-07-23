#!/bin/bash
# task4_p0_mixed_curvature_s21.sh — Phase 2 P0.3 Stage 2.1: Joint Mixed-Curvature RQ-VAE
# 输入: 同 P0.2 (928-dim multi-factor concat), 但用 task4_p0_mixed_curvature_train.py
# 输出: ckpt → task4_p0_mixed_curvature_s22.sh 推断
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

echo "[task58 P0.3] Joint Mixed-Curvature RQ-VAE"
echo "  embedding: multi-factor concat (F_text + F_brand + F_taxonomy + F_behavior = 928-dim)"
echo "  H subspace: first 32 dim (Poincaré ball)"
echo "  E subspace: last 896 dim (Euclidean)"
echo "  max_steps: 3000"

CUDA_VISIBLE_DEVICES=0 /home/wlia0047/ar57_scratch/wenyu/grid_toys/bin/python \
    task_artifacts/scripts/mixed_curvature/task4_p0_mixed_curvature_train.py \
    --max-steps 3000 --batch-size 1024 --gpu 0 --seed 42 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_mixed_curvature_s21.log

echo "[done] ckpt at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task58_p0_mixed_curvature_*/checkpoints/"