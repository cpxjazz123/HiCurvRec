#!/usr/bin/env bash
# Task #320 — Issue #38 Arm C: Regularization (R-Drop alpha=1.0)
# Stage 3 retraining, 200 epoch, Issue #30 SID file, T5-mini 5.5M
# R-Drop = 2 forward passes per batch + symmetric KL → ~50% slower

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task320/armC
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage3_train_${TS}.log"

echo "===== [Task #320 Arm C — R-Drop regularization] launched at $TS =====" | tee "$LOG"
echo "Optimizer: Adam (lr=1e-4), R-Drop: alpha=1.0, 2 forward passes per batch" | tee -a "$LOG"
echo "200 epoch full sweep, seed=42, batch_size=256" | tee -a "$LOG"
echo "Expected wall time: ~140 min (R-Drop 2x forward = 50% slower)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" $REPO/scripts/task320_issue38_5arm_stage3_train.py \
    --arm C --num_epochs 200 --batch_size 256 --gpu 0 --task_id 320 2>&1 | tee -a "$LOG"

echo "===== [Task #320 Arm C] finished at $(date +%Y%m%d_%H%M%S) =====" | tee -a "$LOG"
