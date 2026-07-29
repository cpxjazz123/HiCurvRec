#!/bin/bash
# Task #292 — Restoration (EMA + dead code revival) + κ learnable Stage 1 launcher (cuda:2)
# 2026-07-29

set -u
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

export CUDA_VISIBLE_DEVICES=2
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task292
mkdir -p $TRITON_CACHE_DIR

echo "[Task #292] Stage 1 Restoration + κ learnable 启动 (cuda:2, $(date))"
echo "[Task #292] decay=0.99, revive_threshold=1, revive_ratio=0.1, revive_freq=5 epochs, K=64"

python -u HG-Rec/train_hrqvae.py \
    --data_path HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir ./products/task292/hrqvae_restoration_kappa \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --epochs 200 \
    --batch_size 1024 \
    --lr 1e-3 \
    --beta 0.5 \
    --loss_type poincare \
    --kmeans_init True \
    --eval_step 5 \
    --num_workers 4 \
    --quantizer restoration \
    --codebook_ema_decay 0.99 \
    --revive_threshold 1 \
    --revive_ratio 0.1 \
    --revive_freq_epochs 5 \
    --device cuda:0 \
    --save_limit 1 \
    2>&1 | tee logs/task292/stage1_train.log

echo "[Task #292] Stage 1 完成 (PID $$), 等 Stage 234 waiter fire"
echo "$$" > products/task292/_TRAINING_PID_DONE
