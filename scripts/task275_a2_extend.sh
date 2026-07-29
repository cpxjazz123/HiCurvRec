#!/bin/bash
# Task #275 — A2 extend to ep50 (continue from epoch_29 ckpt)
# 2026-07-29
#
# 背景: A2 ep30 L0=89.1% (borderline < 90% §6.7.4 stop-loss (i) threshold)
#       trajectory 仍上升 (ep25=84.4% → ep30=89.1%, 5 ep +4.7pp)
#       期望 ep50 跨 90%
#
# R7: GPU 1 空闲
# R12: 默认 save_limit=1 + best_collision ckpt
# R8: commit 后立即 push

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=${GPU:-1}

A2_CKPT_DIR=$(ls -td $REPO/products/task270/A2_curriculum/*/ | head -1)
A2_CKPT=$A2_CKPT_DIR/epoch_29_collision_0.1293_model.pth
SAVE_DIR=$REPO/products/task275/A2_extend_ep50
mkdir -p $SAVE_DIR
LOG_FILE=$REPO/logs/task275/A2_extend_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p $REPO/logs/task275

echo "[$(date)] A2 extend 启动"
echo "[$(date)] init_from=$A2_CKPT"
echo "[$(date)] SAVE_DIR=$SAVE_DIR"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task275_a2_extend \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1800 python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 20 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode_list shared,shared,shared \
    --eval_step 5 \
    --save_limit 1 \
    --device cuda:0 \
    --init_encoder_from $A2_CKPT \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
echo "[$(date)] L0 trajectory:"
LOG=$(find $SAVE_DIR -name "hrqvae.log" | head -1)
[ -n "$LOG" ] && grep "step2 monitor ep" "$LOG" | tail -10