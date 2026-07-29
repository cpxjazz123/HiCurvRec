#!/bin/bash
# Task #301 / Issue #30 Gate 2 — Stage 2 Sinkhorn 推断
# 2026-07-29
#
# 背景: Issue #30 Gate 1 PASS (best_collision=0.0873, L0/L1/L2 usage 100%)
#       Gate 2 = Sinkhorn 5 iter inference → (N, 4) SID .npy 给 Stage 3 T5
#       ckpt: products/task301/hrqvae_issue30_gate1/.../best_loss_model.pth
#
# GATE_DECLARATION:
#   gate_2_issue_30: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 2 空闲 (4×L40S, 0/1/3 全部 0%)
# R14: Issue #30 Gate 2 fail → 关闭 issue

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task301
mkdir -p "$LOG_DIR"

export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage2_gate2_${TS}.log

echo "[$(date)] Launching Task #301 / Issue #30 Gate 2: Stage 2 Sinkhorn 5 iter inference"
echo "  ckpt: products/task301/hrqvae_issue30_gate1/.../best_loss_model.pth"
echo "  GPU: 2 (R7 空闲)"
echo "  output: $REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task301_gate2
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task301_issue30_gate2_stage2_codebook.py \
    > "$LOG_FILE" 2>&1 &

INFER_PID=$!
echo "$INFER_PID" > $REPO/products/task301/_INFERENCE_PID_GATE2
echo "[$(date)] Gate 2 Sinkhorn inference PID=$INFER_PID, GPU=2"
echo ""
echo "===== Task #301 / Issue #30 Gate 2 Launched ====="
echo "PID=$INFER_PID (GPU 2, Sinkhorn 5 iter inference)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~2 min"
echo "Gate 2 通过条件: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20"