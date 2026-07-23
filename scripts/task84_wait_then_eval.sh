#!/bin/bash
# Task #84 — wait for Stage 3 to finish + launch Stage 4 test evaluation
# 按 R11.3 + R8 daemon 模式: 自动检测 Stage 3 完成 + 启动 Stage 4

set -e

TRAIN_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task84/_STAGE3_TRAINING_PID
echo "[$(date +%H:%M:%S)] Waiting for Stage 3 (PID file: $TRAIN_PID_FILE)..."

# Wait for PID file to disappear (Stage 3 launch script writes it then rm after training ends)
while [ -f "$TRAIN_PID_FILE" ]; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] Stage 3 done (PID file removed). Sleeping 60s for ckpt flush..."
sleep 60

# Verify best ckpt exists
BEST_CKPT=$(ls -t /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ HG_Rec_best.pth NOT FOUND, exiting"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Found best ckpt: $BEST_CKPT"
echo "[$(date +%H:%M:%S)] Launching Stage 4 test evaluation..."

bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage4_eval.sh
