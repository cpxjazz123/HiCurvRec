#!/bin/bash
# Task #181 (新) — 50 epoch Stage 1 小规模复训, 用于捕捉码字双曲范数 epoch-by-epoch 演化.
# 用户 2026-07-25 诊断假设: Phase 0 修复后, 码字范数从 ~0.1 (无效) 演化到 ~0.85 (强曲率).
# 这次跑专门开 [hypnorm] 日志 (hrqvae_trainer.py 新增), 把每 epoch 三层 stats 落到 log.
#
# GPU 分配: GPU 3 完全空闲 (util 0%, mem 3 MiB). 不抢 #178/#179/#180.
# 复现性: 跟 task180 / 修正 baseline #178 用同样的 β, codebook, sinkhorn 配置.
# 只跑 50 epoch (够看范数曲线), 不存 full ckpt (避免占磁盘).

set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# R7 GPU 分配: GPU 3 完全空闲
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task181_stage1
mkdir -p "$TRITON_CACHE_DIR"

CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_norm_evolution"
mkdir -p "$CKPT_DIR"

python3 HG-Rec/train_hrqvae.py \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir "$CKPT_DIR" \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --beta 0.5 \
    --sk_epsilons 0.003 0.003 0.003 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 100 \
    --epochs 50 \
    --batch_size 1024 \
    --lr 1e-3 \
    --layers 512 256 128 64 \
    --device cuda:0 \
    --save_limit 2