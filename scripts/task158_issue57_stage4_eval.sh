#!/usr/bin/env bash
# Task #158 / Issue #57 Arm A (random init) Stage 4 eval
# Issue #57 Gate 0: 验证 T5 SID token embedding 初始化方式 (random vs hyperbolic) 对 test R@10 的影响
# Decision: > HG-Rec baseline 0.1020 GO vs ≤ 0.1020 NO-GO
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

CKPT_PATH=$(ls -t $REPO/products/task158/ckpt/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"
echo "Modified: $(stat -c '%y' "$CKPT_PATH")"

LOG_DIR=$REPO/logs/task158_stage4
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_eval_armA_${TS}.log"
echo "===== [Task #158 Issue #57 Stage 4 Arm A: random init] launched at $TS =====" | tee "$LOG"

GPU=${1:-0}
echo "GPU $GPU" | tee -a "$LOG"

CODE_PATH="_A2_t5_hrqvae_poincare.npy"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm "A" --ckpt_path "$CKPT_PATH" --beam_size 20 --gpu 0 --task_id 158 \
    --code_path "$CODE_PATH" 2>&1 | tee -a "$LOG"
echo "===== [Task #158 Issue #57 Stage 4 Arm A] completed at $(date) =====" | tee -a "$LOG"