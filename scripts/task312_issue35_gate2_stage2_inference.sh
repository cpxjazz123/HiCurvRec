#!/bin/bash
# Task #312 / Issue #35 Gate 2 — Stage 2 Sinkhorn 推断
# 2026-07-30
#
# 背景: Issue #35 Gate 1 PASS (L0/L1/L2 usage 100%, collision_rate 0.0926)
#       Gate 2 = Sinkhorn 5 iter inference → (N, 4) SID .npy 给 Stage 3 T5
#       ckpt: products/task312/hrqvae_issue35_rl_identity_sl22/.../best_loss_model.pth
#
# GATE_DECLARATION:
#   gate_2_issue_35: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20
#
# R7: GPU 2 空闲 (task309 GPU 0, task307 完成)
# R14: Issue #35 Gate 2 fail → 关闭 issue

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task312
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage2_gate2_${TS}.log

echo "[$(date)] Launching Task #312 / Issue #35 Gate 2: Stage 2 Sinkhorn 5 iter inference"
echo "  ckpt: products/task312/hrqvae_issue35_rl_identity_sl22/.../best_loss_model.pth"
echo "  GPU: 2 (R7 空闲)"
echo "  output: $REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue35_rl_identity_sl22.npy"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task312_gate2
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task312_issue35_gate2_stage2_codebook.py \
    > "$LOG_FILE" 2>&1 &

INFER_PID=$!
echo "$INFER_PID" > $REPO/products/task312/_INFERENCE_PID_GATE2
echo "[$(date)] Gate 2 Sinkhorn inference PID=$INFER_PID, GPU=2"
echo ""
echo "===== Task #312 / Issue #35 Gate 2 Launched ====="
echo "PID=$INFER_PID (GPU 2, Sinkhorn 5 iter inference)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~2 min"
echo "Gate 2 通过条件: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20"
