#!/bin/bash
# Task #350 / Issue #62 Gate 2 — Stage 2 Sinkhorn 推断 (Arm C #30+#43 联合)
# 2026-07-31
#
# 背景: Issue #62 §Gate 1 Arm C (#30 + #43 联合, baseline K + Sinkhorn ON) Stage 1 PASS
#       Stage 2 = Sinkhorn 5 iter inference → (N, 4) SID .npy 给 Stage 3 T5
#       ckpt: products/task350/hrqvae_issue62_gate1_armc/.../best_loss_model.pth
#
# GATE_DECLARATION (mirror Issue #30 Gate 2):
#   gate_2_issue_62: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20
#
# R7: GPU 0 空闲 (4×L40S, Arm C Stage 1 已完成, GPU 0 释放)
# R14: Issue #62 hard-stop at Gate 2 fail

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task350
mkdir -p "$LOG_DIR"

export PYTHONPATH=/home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages:$REPO/HG-Rec:${PYTHONPATH:-}

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage2_gate2_${TS}.log

echo "[$(date)] Launching Task #350 / Issue #62 Gate 2: Stage 2 Sinkhorn 5 iter inference" | tee "$LOG_FILE"
echo "  ckpt: $REPO/products/task350/hrqvae_issue62_gate1_armc/*/best_loss_model.pth" | tee -a "$LOG_FILE"
echo "  output: $REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue62_gate1_armc.npy" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python \
  $REPO/scripts/task350_issue62_gate2_armc_stage2_codebook_v3.py 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Task #350 / Issue #62 Gate 2 Arm C Stage 2 inference completed" | tee -a "$LOG_FILE"