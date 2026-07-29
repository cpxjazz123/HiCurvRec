#!/bin/bash
# Task #326 / K=384 sweet spot probe — Stage 1 (R12 强制 ckpt)
# K0=384 + L1=128 + L2=256 (跟 task194 K=256 一致架构, 仅 K0 替换)
# GPU 1 (R7 等待 task320 释放), 200 epoch 跟 task194 baseline 一致

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task326
mkdir -p "$LOG_DIR"

export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task326/hrqvae_k0_384
mkdir -p $SAVE_PATH

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_k0_384_${TS}.log

echo "[$(date)] Launching Task #326 K=384 Stage 1"
echo "  --num_emb_list 384 128 256 (K0=384 probe)"
echo "  GPU: 1 (R7 等待 task320 释放)"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task326_k384
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/HG-Rec/train_hrqvae.py \
    --lr 1e-3 \
    --epochs 200 \
    --batch_size 512 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 5 \
    --data_path $REPO/HG-Rec/dataset/Instruments/item_emb.parquet \
    --weight_decay 0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 384 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task326/_TRAINING_PID_STAGE1
echo "[$(date)] K=384 Stage 1 PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #326 / K=384 Stage 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, K0=384, 200 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-1.5h"