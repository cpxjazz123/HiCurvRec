#!/bin/bash
# Task #144 — κ + codebook 解耦训练调度 (用户 2026-07-24 提议)
# 2026-07-24
#
# 验证用户提议的 2 阶段训练调度:
#   Phase A: κ 冻结在 0, codebook 单独训练到健康稳定
#   Phase B: 解冻 κ + 极小 lr_theta, 监控 utilization, 掉则 freeze
#
# 2 臂并行:
#   Arm A (Phase A only): 200 epoch 全程 κ 冻结, 验证 baseline stability
#   Arm B (Phase A + Phase B): 100 ep Phase A + 100 ep Phase B, 1e-5 lr_theta
#
# R7: GPU 0 (killed Task #140) + GPU 2 (killed Task #136) 空闲.
#     GPU 1 occupied by Task #143 FDSA (~30-60 min ETA), 暂不并行第 3 臂.
# R12: best_loss auto-save per epoch.
# R137: NaN guard 已嵌入 (触发即 raise).

set -uo pipefail
cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task144
mkdir -p "$LOG_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# ===== Arm A: Phase A only (200 ep, 全程 κ 冻结) =====
PRODUCTS_DIR_A=/home/wlia0047/ar57/wenyu/GeneRec/products/task144/train/arm_A_phaseA_only
mkdir -p "$PRODUCTS_DIR_A"
TS_A=$(date +%Y%m%d_%H%M%S)
LOG_A="$LOG_DIR/task144_arm_A_${TS_A}.log"

echo "[$(date)] Launching Task #144 Arm A: Phase A only (200 ep, κ frozen) — log: $LOG_A"

# Triton cache isolation (per CLAUDE.md GPU table, sm_89 L40S)
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task144_arm_A
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=200 \
    --utilization_freeze_threshold=1.0 \
    --epochs=200 \
    --seed=42 \
    --ckpt_dir="$PRODUCTS_DIR_A" \
    --kappa_log_path="$PRODUCTS_DIR_A/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR_A/phase_a_baseline.json" \
    > "$LOG_A" 2>&1 &

TRAIN_PID_A=$!
echo "$TRAIN_PID_A" > /home/wlia0047/ar57/wenyu/GeneRec/products/task144/_TRAINING_PID_A
echo "[$(date)] Arm A PID=$TRAIN_PID_A, ckpt_dir=$PRODUCTS_DIR_A"

# ===== Arm B: Phase A (100 ep) + Phase B (100 ep with 1e-5 lr_theta) =====
PRODUCTS_DIR_B=/home/wlia0047/ar57/wenyu/GeneRec/products/task144/train/arm_B_decouple
mkdir -p "$PRODUCTS_DIR_B"
TS_B=$(date +%Y%m%d_%H%M%S)
LOG_B="$LOG_DIR/task144_arm_B_${TS_B}.log"

echo "[$(date)] Launching Task #144 Arm B: Phase A 100ep + Phase B 100ep (lr_theta=1e-5, freeze-on-collapse 5%) — log: $LOG_B"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task144_arm_B
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=100 \
    --lr_theta_post_unfreeze=1e-5 \
    --utilization_freeze_threshold=0.05 \
    --epochs=200 \
    --seed=42 \
    --ckpt_dir="$PRODUCTS_DIR_B" \
    --kappa_log_path="$PRODUCTS_DIR_B/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR_B/phase_a_baseline.json" \
    > "$LOG_B" 2>&1 &

TRAIN_PID_B=$!
echo "$TRAIN_PID_B" > /home/wlia0047/ar57/wenyu/GeneRec/products/task144/_TRAINING_PID_B
echo "[$(date)] Arm B PID=$TRAIN_PID_B, ckpt_dir=$PRODUCTS_DIR_B"

echo ""
echo "===== Task #144 Launched ====="
echo "Arm A PID=$TRAIN_PID_A (GPU 0, Phase A only)"
echo "Arm B PID=$TRAIN_PID_B (GPU 2, Phase A 100 + Phase B 100)"