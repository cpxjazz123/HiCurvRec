#!/bin/bash
# Task #300 / Issue #29 Gate 1 — Stage 1 100 epoch 训练 (per-layer K_l + per-layer c_k range)
# 2026-07-29
#
# 背景: Issue #29 §Gate 0 PASS (scripts/task300_issue29_gate0_per_layer_k.py)
#       Gate 1 = Stage 1 100 epoch 训练 per-layer K_l=[128,64,32] + per-layer c_k range
#       使用 baseline train_hrqvae.py (Issue #29 沿用 baseline HRQVAE 构造)
#
# GATE_DECLARATION:
#   gate_1_issue_29: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 0/1 空闲 (4×L40S, 当前 0% util 0 MB used per nvidia-smi)
# R12: save_limit=1 + 每个 epoch 强制保存
# R14: Issue #29 hard-stop at Gate 1 fail

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task300
mkdir -p "$LOG_DIR"

export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task300/hrqvae_issue29_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #300 / Issue #29 Gate 1: Stage 1 100 epoch per-layer K_l"
echo "  --num_emb_list 128 64 32 (L0 K_l 翻倍, L1/L2 K_l 减半)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (per-layer 异构 c_k range, task242 Arm A)"
echo "  --gumbel_tau_l '0.0,0.0,0.0' (关闭 Issue #28 Gumbel, Issue #29 沿用 baseline argmin)"
echo "  GPU: 0 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task300_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/HG-Rec/train_hrqvae.py \
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
    --bn False \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 128 64 32 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --euclidean_qloss False \
    --loss_mult_codebook 1.0 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --output_dir $SAVE_PATH/ \
    --c_k_range_list "1.0:5.0,0.5:20.0,0.5:20.0" \
    --c_k_min 0.5 \
    --c_k_max 20.0 \
    --c_k_seed 42 \
    --gumbel_tau_l "0.0,0.0,0.0" \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task300/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=0"
echo ""
echo "===== Task #300 / Issue #29 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, per-layer K_l, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h (HRQ-VAE 9.18M × 100 epoch)"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50"
