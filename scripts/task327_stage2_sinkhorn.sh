#!/bin/bash
# Task #327 / K=256 + Issue #30 — Stage 2 Sinkhorn codebook generation
# 输入: Stage 1 best_collision_model.pth
# 输出: Instruments_t5_rqvae_k256_issue30.npy (9922, 4) SID

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task327
mkdir -p "$LOG_DIR"

CKPT=$(find $REPO/products/task327 -name "best_collision_model.pth" | head -1)
SID_OUT=$REPO/products/task327/Instruments_t5_rqvae_k256_issue30.npy
LOG_FILE=$LOG_DIR/stage2_sinkhorn_$(date +%Y%m%d_%H%M%S).log

if [ ! -f "$CKPT" ]; then
    echo "❌ no best_collision_model.pth found" >&2
    exit 1
fi

echo "[$(date)] Task #327 Stage 2 Sinkhorn: ckpt=$CKPT"
echo "[$(date)] Output: $SID_OUT"

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u $REPO/scripts/task194_stage2_codebook.py \
    --ckpt_path "$CKPT" \
    --output_path "$SID_OUT" \
    --device cuda:0 \
    --max_sinkhorn_iters 30 \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✅ Stage 2 Sinkhorn done: $SID_OUT"
    ls -la "$SID_OUT"
else
    echo "[$(date)] ❌ Stage 2 failed (exit=$EXIT_CODE)"
    tail -30 "$LOG_FILE"
fi