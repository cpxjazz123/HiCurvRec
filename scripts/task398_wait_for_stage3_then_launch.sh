#!/bin/bash
# Task #398 wait_for_stage3 helper — 等待 task396b Stage 3 T5 训练完成, 然后自动 launch Stage 4 R@K eval
# Usage: bash scripts/task398_wait_for_stage3_then_launch.sh [gpu_id]

set -e

GPU_ID=${1:-1}
STAGE3_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/_STAGE3_TRAINING_PID
MAX_WAIT_MIN=200  # Stage 3 预计 ~143 min, 给 60 min buffer
CHECK_INTERVAL=60  # 每 60 sec 检查一次

if [ ! -f "$STAGE3_PID_FILE" ]; then
    echo "❌ Stage 3 PID file missing: $STAGE3_PID_FILE"
    exit 1
fi

STAGE3_PID=$(cat $STAGE3_PID_FILE)
echo "[Task #398 wait] Stage 3 PID = $STAGE3_PID, GPU for Stage 4 = $GPU_ID"
echo "[Task #398 wait] Max wait = $MAX_WAIT_MIN min, check interval = $CHECK_INTERVAL sec"

ELAPSED=0
while [ $ELAPSED -lt $((MAX_WAIT_MIN * 60)) ]; do
    if ! ps -p $STAGE3_PID > /dev/null 2>&1; then
        echo "[Task #398 wait] Stage 3 process $STAGE3_PID finished after $ELAPSED sec"
        break
    fi
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    LATEST_LOG=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/logs/task396b_t5mini_stage3_*.log 2>/dev/null | head -1)
    EPOCH=$(grep -oE "Epoch [0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1)
    BATCH=$(grep -oE "[0-9]+/515" "$LATEST_LOG" 2>/dev/null | tail -1)
    echo "[Task #398 wait] +$ELAPSED sec | Stage 3 still running | $EPOCH | $BATCH"
done

if ps -p $STAGE3_PID > /dev/null 2>&1; then
    echo "❌ Stage 3 still running after $MAX_WAIT_MIN min — manual check needed"
    exit 1
fi

# Verify ckpt exists
BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task396_issue99_stage3_t5_train/ckpt/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 ckpt MISSING — R12 violation"
    exit 1
fi
echo "[Task #398 wait] Stage 3 ckpt OK: $BEST_CKPT"

# Launch Stage 4 eval on GPU $GPU_ID
echo "[Task #398 wait] Launching Stage 4 eval on GPU $GPU_ID..."
cd /home/wlia0047/ar57/wenyu/GeneRec
export CUDA_VISIBLE_DEVICES=$GPU_ID
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task398_stage4_rk_eval.sh