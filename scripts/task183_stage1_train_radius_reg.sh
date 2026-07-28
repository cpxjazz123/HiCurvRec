#!/bin/bash
# Task #183 Phase 1 v2 — 半径正则化 RQ-VAE — Stage 1 训练
# GPU target: TBD (R7 check)
#
# 执行: bash scripts/task183_stage1_train_radius_reg.sh

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task183/hrqvae_radius_reg
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task183

export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task183
mkdir -p $TRITON_CACHE_DIR

GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader)
echo "GPU 状态:
$GPUS"

echo "=========================================="
echo "Task #183 Phase 1 v2 Stage 1 启动时间: $(date)"
echo "半径正则化: radii=[1.0, 1.345, 1.69], rho_reg_weight=0.1"
echo "Phase 0.6 官方 loss + argmin (Sinkhorn OFF)"
echo "=========================================="

python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --radii 1.0 1.345 1.69 \
    --rho_reg_weight 0.1 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task183/hrqvae_radius_reg \
    --device cuda:0 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task183/stage1_radius_reg.out

echo "=========================================="
echo "Task #183 Phase 1 v2 Stage 1 结束时间: $(date)"
echo "=========================================="
