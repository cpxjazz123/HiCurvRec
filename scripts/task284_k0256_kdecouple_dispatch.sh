#!/bin/bash
# Task #284 — Issue #10 follow-up: 用 task194_k0256 SID + κ-decouple 3-arm 验证
# 2026-07-29
#
# 复用 task144 调度 (Arm A κ frozen + Arm B κ unfreeze), 但 num_emb_list 改 K=256
#   (task194_k0256 R@10=0.1053 当前最佳, 验证 κ-decouple 是否能在最优 K 下突破)
#
# 2 臂并行 (Arm A → GPU 2, Arm B → GPU 3, GPU 0/1 留给其他任务)
# Arm C = task194_k0256 复用 (Stage 3+4 已跑过, 直接用 R@10=0.1053 作为对照)
#
# R7: GPU 0/1 完全空闲, GPU 2/3 完全空闲 → 2 臂并行无冲突
# R12: best_loss auto-save per epoch
# R137: NaN guard 已嵌入 (触发即 raise)

set -uo pipefail
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task284
mkdir -p "$LOG_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# ===== Arm A: Phase A only (200 ep, 全程 κ 冻结) =====
PRODUCTS_DIR_A=$REPO/products/task284/hrqvae_k0256_armA_phaseA_only
mkdir -p "$PRODUCTS_DIR_A"
TS_A=$(date +%Y%m%d_%H%M%S)
LOG_A="$LOG_DIR/stage1_armA_${TS_A}.log"

echo "[$(date)] Launching Task #284 Arm A: Phase A only (200 ep, κ frozen, K=256) — log: $LOG_A"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armA
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=200 \
    --utilization_freeze_threshold=1.0 \
    --epochs=200 \
    --seed=42 \
    --num_emb_list 256 128 256 \
    --batch_size=256 \
    --ckpt_dir="$PRODUCTS_DIR_A" \
    --kappa_log_path="$PRODUCTS_DIR_A/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR_A/phase_a_baseline.json" \
    > "$LOG_A" 2>&1 &

TRAIN_PID_A=$!
echo "$TRAIN_PID_A" > $REPO/products/task284/_TRAINING_PID_A
echo "[$(date)] Arm A PID=$TRAIN_PID_A, ckpt_dir=$PRODUCTS_DIR_A, GPU=2"

# ===== Arm B: Phase A (100 ep) + Phase B (100 ep with 1e-5 lr_theta) =====
PRODUCTS_DIR_B=$REPO/products/task284/hrqvae_k0256_armB_decouple
mkdir -p "$PRODUCTS_DIR_B"
TS_B=$(date +%Y%m%d_%H%M%S)
LOG_B="$LOG_DIR/stage1_armB_${TS_B}.log"

echo "[$(date)] Launching Task #284 Arm B: Phase A 100ep + Phase B 100ep (lr_theta=1e-5, freeze-on-collapse 5%, K=256) — log: $LOG_B"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task284_armB
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=100 \
    --lr_theta_post_unfreeze=1e-5 \
    --utilization_freeze_threshold=0.05 \
    --epochs=200 \
    --seed=42 \
    --num_emb_list 256 128 256 \
    --batch_size=256 \
    --ckpt_dir="$PRODUCTS_DIR_B" \
    --kappa_log_path="$PRODUCTS_DIR_B/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR_B/phase_a_baseline.json" \
    > "$LOG_B" 2>&1 &

TRAIN_PID_B=$!
echo "$TRAIN_PID_B" > $REPO/products/task284/_TRAINING_PID_B
echo "[$(date)] Arm B PID=$TRAIN_PID_B, ckpt_dir=$PRODUCTS_DIR_B, GPU=3"

echo ""
echo "===== Task #284 Launched ====="
echo "Arm A PID=$TRAIN_PID_A (GPU 2, Phase A only, K=256)"
echo "Arm B PID=$TRAIN_PID_B (GPU 3, Phase A 100 + Phase B 100, K=256)"
echo "Arm C = task194_k0256 reuse (R@10=0.1053 baseline)"
echo "Monitor: tail -f $LOG_A and $LOG_B"
echo "Stage 2 + Stage 3 + Stage 4 待 Stage 1 完成后手动 fire"
