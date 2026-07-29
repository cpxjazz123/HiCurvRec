#!/usr/bin/env bash
# Task #320 Arm C Stage 4 eval @ beam=100
# Generic Stage 4 eval launcher — runs after training completes
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

# Find latest ckpt for arm C
CKPT_PATH=$(ls -t $REPO/products/task320/armC/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found in $REPO/products/task320/armC/"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"

LOG_DIR=$REPO/logs/task320/armC
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_beam100_${TS}.log"
echo "===== [Task #320 Arm C beam=100] launched at $TS =====" | tee "$LOG"

GPU=${1:-0}
echo "GPU $GPU" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm C --ckpt_path "$CKPT_PATH" --beam_size 100 --gpu 0 --task_id 320 2>&1 | tee -a "$LOG"
echo "===== [Task #320 Arm C beam=100] completed at $(date) =====" | tee -a "$LOG"
