#!/usr/bin/env bash
# Task #320 — Issue #38 Arm A: Optimizer 改造 (AdamW + cosine + warmup 5000 + dropout 0.05)
# Stage 3 retraining, 200 epoch, Issue #30 SID file, T5-mini 5.5M

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task320/armA
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage3_train_${TS}.log"

echo "===== [Task #320 Arm A — Optimizer 改造] launched at $TS =====" | tee "$LOG"
echo "Optimizer: AdamW (lr=1e-3, wd=0.05), Scheduler: cosine + warmup 5000, Dropout: 0.05" | tee -a "$LOG"
echo "200 epoch full sweep, seed=42, batch_size=256" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" $REPO/scripts/task320_issue38_5arm_stage3_train.py \
    --arm A --num_epochs 200 --batch_size 256 --gpu 0 --task_id 320 2>&1 | tee -a "$LOG"

echo "===== [Task #320 Arm A] finished at $(date +%Y%m%d_%H%M%S) =====" | tee -a "$LOG"
