#!/bin/bash
# Task #194 / Issue #40 Gate 1 — Stage 4 eval (protocol-matched K=64 control)
# task194 Stage 3 finished Ep 56/200 (early terminated by external event, best ckpt preserved at Ep 55)
# best ckpt: products/task194/protocol_match_k064/t5mini_k064/Instruments/Jul-30-2026_13-59-42/HG_Rec_best.pth
# best val_NDCG@20=0.0983, val_R@10=0.1258 (at Ep 55)
# Decision threshold (per Issue #40 Gate 0 verdict): test_R@10 vs HG-Rec baseline 0.1020
#   - test_R@10 ≤ 0.1020 → task194 protocol leak CONFIRMED (task194 K=64 not a real lever)
#   - test_R@10 > 0.1020 → task194 K=64 protocol 真有效, 升级查 #84 baseline Stage 1/2 bug
#   - test_R@10 ≈ 0.1041+ → significant, baseline reversal candidate
# Default fail-safe: beam=20 (proven baseline, low GPU mem)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

GPU=${1:-2}
BEAM=${2:-20}

CKPT_PATH=$REPO/products/task194/protocol_match_k064/t5mini_k064/Instruments/Jul-30-2026_13-59-42/HG_Rec_best.pth
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: ckpt missing at $CKPT_PATH"
    exit 1
fi
echo "Using ckpt: $CKPT_PATH"
echo "Modified: $(stat -c '%y' $CKPT_PATH)"

LOG_DIR=$REPO/logs/task194_gate1/stage4
mkdir -p $LOG_DIR

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage4_beam${BEAM}_${TS}.log"
echo "===== [Task #194 K=64 protocol-matched Stage 4 K=$BEAM eval] launched at $TS =====" | tee "$LOG"
echo "GPU $GPU" | tee -a "$LOG"

CODE_PATH="_t5_hrqvae_protocol_match_k064.npy"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" $REPO/scripts/task320_arm_stage4_eval.py \
    --arm "A" --ckpt_path "$CKPT_PATH" --beam_size $BEAM --gpu 0 --task_id 194 \
    --code_path "$CODE_PATH" 2>&1 | tee -a "$LOG"
echo "===== [Task #194 K=64 protocol-matched Stage 4 K=$BEAM eval] completed at $(date) =====" | tee -a "$LOG"