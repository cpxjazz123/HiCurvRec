#!/bin/bash
# Task #312 — Issue #35 r_l=[1,1,1] identity + s_l=[2,2,2] Stage 1 100 epoch 训练
# 目的: 验证 s_l 是不是真杠杆 (isolate r_l 边际效应)
# R11.5 决策: r_l identity 移除 r_l 极端值杠杆, 保留 s_l=[2,2,2] (跟 #30 一致)
# 关联: Issue #35 (OPEN) + task304 D6 ablation + task301 #30 GO
# 2026-07-30

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task312
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task312/hrqvae_issue35_rl_identity_sl22
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #312 / Issue #35 r_l=[1,1,1] + s_l=[2,2,2] Stage 1"
echo "  --radius_list 1.0 1.0 1.0 (r_l identity baseline)"
echo "  --scale_list 2.0 2.0 2.0 (#30 GO per-layer scale)"
echo "  GPU: 2 (R7 空闲, task309 Stage 3 用 GPU 0, task307 已完成)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task312_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task312_issue35_rl_identity_sl22_stage1_train.py \
    --lr 1e-3 \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path $REPO/HG-Rec/dataset/Instruments/item_emb.parquet \
    --weight_decay 0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task312/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=2"
echo ""
echo "===== Task #312 / Issue #35 r_l identity + s_l=[2,2,2] Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 2, r_l identity + s_l GO, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20"
