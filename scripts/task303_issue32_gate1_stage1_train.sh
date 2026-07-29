#!/bin/bash
# Task #303 / Issue #32 Gate 1 — Stage 1 100 epoch 训练 (per-layer Codebook Transforms + c_k range 双轴)
# 2026-07-30
#
# 背景: Issue #32 Gate 0 PASS
#       Gate 1 = Stage 1 100 epoch 训练 per-layer r_l + s_l + per-layer c_k range (双轴协同)
#       ckpt_dir = products/task303/hrqvae_issue32_gate1/
#
# GATE_DECLARATION:
#   gate_1_issue_32: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 0 空闲 (Issue #29/#30 Stage 3 已完成释放)
# R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth
# R14: Issue #32 hard-stop at Gate 1 fail

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task303
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

SAVE_PATH=$REPO/products/task303/hrqvae_issue32_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #303 / Issue #32 Gate 1: Stage 1 100 epoch"
echo "  --radius_list 0.5 1.0 2.0 (per-layer 半径缩放, 中间值)"
echo "  --scale_list 1.0 1.0 1.0 (per-layer scale factor)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (per-layer c_k range 双轴协同)"
echo "  GPU: 0 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task303_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task303_issue32_gate1_stage1_train.py \
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
    --loss_mult_codebook 1.0 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    --radius_list 0.5 1.0 2.0 \
    --scale_list 1.0 1.0 1.0 \
    --c_k_range_list "1.0:5.0,0.5:20.0,0.5:20.0" \
    --c_k_seed 42 \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task303/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=0"
echo ""
echo "===== Task #303 / Issue #32 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, per-layer Codebook Transforms + c_k range 双轴, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~30-60 min"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50"