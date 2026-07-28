#!/bin/bash
# Task #181 — 100% 官方对齐 Stage 1 RQ-VAE 训练 (Phase 0.6)
# 用户 2026-07-25 指令 "100% 对齐官方".
# 配置: batch_size=1024, beta=1.0 (β on codebook), 还原官方 loss + logmap0,
#       sk_epsilons=[0,0,0] (官方默认关闭), epochs=1000, kmeans_init + 1000 iters
# GPU 1 (空闲, nvidia-smi 已确认).
# 必须用 grid_toys env (含 torch 2.6.0+cu124), 否则 ModuleNotFoundError.
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task181

# Triton cache 强制为 sm_89 L40S fresh compile (避免 sm_86 旧 kernel 加载失败)
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task181
mkdir -p $TRITON_CACHE_DIR

START_TIME=$(date +%s)
echo "=========================================="
echo "Task #181 Stage 1 启动时间: $(date)"
echo "GPU target: cuda:1"
echo "官方 0bcfd0f 对齐 (Poincaré loss on raw vectors + logmap0 + β=1.0 on codebook)"
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
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2 \
    --device cuda:1 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task181/stage1_train.out

END_TIME=$(date +%s)
echo "=========================================="
echo "Task #181 Stage 1 结束时间: $(date)"
echo "耗时: $((END_TIME - START_TIME)) 秒 ($(( (END_TIME - START_TIME) / 60 )) 分钟)"
echo "=========================================="