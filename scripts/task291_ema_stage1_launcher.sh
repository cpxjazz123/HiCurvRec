#!/bin/bash
# Task #291 — EMA codebook + κ learnable Stage 1 launcher (cuda:1)
# 2026-07-29

set -u
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task291
mkdir -p $TRITON_CACHE_DIR

echo "[Task #291] Stage 1 EMA codebook + κ learnable 启动 (cuda:1, $(date))"
echo "[Task #291] decay=0.99, K=64, epochs=200 (no kappa_freeze)"

python -u HG-Rec/train_hrqvae.py \
    --data_path HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir ./products/task291/hrqvae_ema_kappa \
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
    --quantizer ema \
    --codebook_ema_decay 0.99 \
    --device cuda:0 \
    --save_limit 1 \
    2>&1 | tee logs/task291/stage1_train.log

echo "[Task #291] Stage 1 完成 (PID $$), 等 Stage 234 waiter fire"
echo "$$" > products/task291/_TRAINING_PID_DONE
