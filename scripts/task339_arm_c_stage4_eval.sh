#!/usr/bin/env bash
# Task #339 / Issue #52 arm_c Stage 4 eval (phase_b_epochs 500)
# Stage 3 terminated ~20:39, ckpt saved. Decision threshold: > 0.1020 GO vs HG-Rec baseline.
# R7: GPU 3 (free since arm_c Stage 3 ended)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$(ls -t $REPO/products/task339/arm_c/t5_out/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"
echo "Modified: $(stat -c '%y' "$CKPT_PATH")"

LOG_DIR=$REPO/logs/task339_arm_c
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_${TS}.log"
echo "===== [Task #339 arm_c Stage 4 eval] launched at $TS =====" | tee "$LOG"

GPU=${1:-3}
echo "GPU $GPU" | tee -a "$LOG"

CODE_PATH="_t5_rqvae_task339_arm_c.npy"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm "C" --ckpt_path "$CKPT_PATH" --beam_size 20 --gpu 0 --task_id 339 \
    --code_path "$CODE_PATH" 2>&1 | tee -a "$LOG"
echo "===== [Task #339 arm_c Stage 4 eval] completed at $(date) =====" | tee -a "$LOG"