#!/bin/bash
# Task #290 — FSQ + κ-decouple Stage 1 launcher (cuda:0)
# 2026-07-29

set -u
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# GPU 绑定
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task290
mkdir -p $TRITON_CACHE_DIR

# Stage 1 FSQ + κ-decouple training (200 epoch, K=64 baseline, FSQ levels [8,5,5,5])
echo "[Task #290] Stage 1 FSQ + κ-decouple 启动 (cuda:0, $(date))"
echo "[Task #290] FSQ levels=[8,5,5,5], K=64, epochs=200, kappa_freeze_epochs=100"

python -u HG-Rec/train_hrqvae.py \
    --data_path HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir ./products/task290/hrqvae_fsq_kappa_decouple \
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
    --quantizer fsq \
    --fsq_levels 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 \
    --kappa_freeze_epochs 100 \
    --lr_theta_post_unfreeze 1e-5 \
    --device cuda:0 \
    --save_limit 1 \
    2>&1 | tee logs/task290/stage1_train.log

# R12 强制: best ckpt 保留 (save_limit=1 + 自带 best_loss_model.pth)
echo "[Task #290] Stage 1 完成 (PID $$), 等 Stage 234 waiter fire"
echo "$$" > products/task290/_TRAINING_PID_DONE
