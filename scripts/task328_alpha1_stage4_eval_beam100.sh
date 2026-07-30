#!/bin/bash
# Task #328 R-Drop α=1.0 Stage 4 K=100 eval (Issue #30 + Issue #38 Layer 2 leader)
# Decision threshold (R10 NORTH STAR): Test R@10 > 0.1053 (task194 K=256 ceiling) → GO
# Default fail-safe ckpt path = α=1.0 best (val_NDCG@20=0.0965, val_R@10=0.1225)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

ALPHA=${1:-1.0}
GPU=${2:-2}

# Map alpha to arm letter (A=1.0, B=2.0, C=0.5, D=4.0)
case $ALPHA in
    0.5) ARM="C" ;;
    1.0) ARM="A" ;;
    2.0) ARM="B" ;;
    4.0) ARM="D" ;;
    *) ARM="E" ;;
esac

# Find best ckpt for the requested alpha
CKPT_PATH=$(ls -t $REPO/products/task328_rdrop_alpha_sweep/alpha_${ALPHA}/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: no ckpt found for α=$ALPHA in $REPO/products/task328_rdrop_alpha_sweep/alpha_${ALPHA}/"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH (alpha=$ALPHA, arm=$ARM)"
echo "Modified: $(stat -c '%y' $CKPT_PATH)"

LOG_DIR=$REPO/logs/task328_rdrop_alpha_sweep/stage4
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_alpha${ALPHA}_arm${ARM}_beam100_${TS}.log"
echo "===== [Task #328 R-Drop α=$ALPHA (Arm $ARM) K=100 eval] launched at $TS =====" | tee "$LOG"
echo "GPU $GPU" | tee -a "$LOG"

CODE_PATH="_t5_rqvae_k256_issue30.npy"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm "$ARM" --ckpt_path "$CKPT_PATH" --beam_size 100 --gpu 0 --task_id 328 \
    --code_path "$CODE_PATH" 2>&1 | tee -a "$LOG"
echo "===== [Task #328 R-Drop α=$ALPHA K=100 eval] completed at $(date) =====" | tee -a "$LOG"
