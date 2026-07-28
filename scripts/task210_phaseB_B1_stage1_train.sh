#!/bin/bash
# Task #210 Phase B1 — product_manifold (无 path reg) Stage 1 RQ-VAE 训练.
# 基于 Phase A 推荐: d_hyp=8, ρ=3.0, e_dim=40 (8 hyp + 32 euc).
# 4 臂设计 (用户 2026-07-26 提案):
#   B0: A0 baseline (#181, 不重训, 复用)
#   B1: product_manifold only ← 本 launcher
#   B2: B1 + path_reg hyp
#   B3: B1 + path_reg euc control
# GPU 0 (R7: A3 在 GPU 1, 0/2/3 空闲).
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task210/hrqvae_B1
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task210

# Triton cache 强制为 sm_89 L40S fresh compile (避免 sm_86 旧 kernel 加载失败)
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task210_B1
mkdir -p $TRITON_CACHE_DIR

# R12: PID 记录
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task210/_TRAINING_PID_B1

START_TIME=$(date +%s)
echo "=========================================="
echo "Task #210 Phase B1 启动时间: $(date)"
echo "GPU target: cuda:0"
echo "Config: product_manifold=True, angular_dim=8, radial_dim=32, e_dim=40"
echo "        num_emb_list=[64,128,256], c=1.0 (default fixed)"
echo "        β=1.0, sk_epsilons=[0,0,0], kmeans_init=True, 1000 iters"
echo "        epochs=1000, batch_size=1024, lr=1e-3"
echo "=========================================="

python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 40 \
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
    --product_manifold \
    --angular_dim 8 \
    --radial_dim 32 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task210/hrqvae_B1 \
    --device cuda:0 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task210/B1_stage1_train.out &
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE
echo "B1 PID: $TRAIN_PID (saved to $PID_FILE)"

wait $TRAIN_PID

END_TIME=$(date +%s)
echo "=========================================="
echo "Task #210 Phase B1 结束时间: $(date)"
echo "耗时: $((END_TIME - START_TIME)) 秒 ($(( (END_TIME - START_TIME) / 60 )) 分钟)"
echo "=========================================="

# R12: 训练结束后清 PID
rm -f $PID_FILE
