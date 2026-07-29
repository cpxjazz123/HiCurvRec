#!/usr/bin/env bash
# Task #320 — Issue #38 Arm E: control (Adam + constant LR + dropout 0.1 + FP32)
# Stage 3 retraining, 200 epoch, Issue #30 SID file, T5-mini 5.5M
# = baseline recipe for fair 5-arm comparison

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task320/armE
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage3_train_${TS}.log"

echo "===== [Task #320 Arm E — control baseline] launched at $TS =====" | tee "$LOG"
echo "Optimizer: Adam (lr=1e-4), Scheduler: constant, Dropout: 0.1, FP32" | tee -a "$LOG"
echo "200 epoch full sweep, seed=42, batch_size=256" | tee -a "$LOG"

# Use GPU 3 (last free GPU after A/B/D on 0/1/3)
CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" $REPO/scripts/task320_issue38_5arm_stage3_train.py \
    --arm E --num_epochs 200 --batch_size 256 --gpu 0 --task_id 320 2>&1 | tee -a "$LOG"

echo "===== [Task #320 Arm E] finished at $(date +%Y%m%d_%H%M%S) =====" | tee -a "$LOG"
