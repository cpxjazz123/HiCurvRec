#!/usr/bin/env bash
# Task #340 / Issue #53 arm_a Stage 4 eval (硬性两阶段 κ 解冻, control = Issue #49)
# Stage 3 terminated ~22:06, ckpt saved. Decision threshold: > 0.1020 GO vs HG-Rec baseline.
# R7: GPU 0 (40% util only, fits idle)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$(ls -t $REPO/products/task340/arm_a/t5_out/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"
echo "Modified: $(stat -c '%y' "$CKPT_PATH")"

LOG_DIR=$REPO/logs/task340_arm_a
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_${TS}.log"
echo "===== [Task #340 arm_a Stage 4 eval] launched at $TS =====" | tee "$LOG"

GPU=${1:-0}
echo "GPU $GPU" | tee -a "$LOG"

CODE_PATH="_t5_rqvae_task340_arm_a.npy"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm "A" --ckpt_path "$CKPT_PATH" --beam_size 20 --gpu 0 --task_id 340 \
    --code_path "$CODE_PATH" 2>&1 | tee -a "$LOG"
echo "===== [Task #340 arm_a Stage 4 eval] completed at $(date) =====" | tee -a "$LOG"