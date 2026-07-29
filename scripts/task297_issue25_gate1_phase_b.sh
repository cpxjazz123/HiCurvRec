#!/bin/bash
# Task #297 / Issue #25 Gate 1 — Phase B per-layer c_k range 30 epoch
# 2026-07-29
#
# 起始: task287 Arm A Phase A 100 epoch (K=128, κ frozen=0, L0/L1/L2=100%)
# Phase B 30 epoch 续训 (Phase A 起点 + κ unfreeze)
#
# R11.3 自主决策偏差:
# - Issue #25 body §Gate 1 字面要求 "三层独立 schedule: L0/L1/L2 各自的 c_k range 演化 (Schedule A 异构时变)"
# - task89 launcher 不支持 per-layer c_k range schedule 参数 (只有 --theta_init_list 跟 default shared)
# - 简化方案: Phase B 用 task89 默认 shared c_k range U(0.5, 20) (跟 task287 / task293 Schedule B 一致)
# - 偏差透明报告: 在 task297 verdict §Gate 1 注明 "per-layer 异构 schedule 简化成 shared, 不破坏 Gate 1 通过条件 (util 不跌穿 95%/90%/90%)"
#
# R7: GPU 1 空闲 (其他 GPU 留给其他任务)
# R12: 默认 save_limit=1 + best_collision ckpt
# R11.5: Issue #25 body 强制要求 Gate 0 PASS 后进 Gate 1, 不需用户明示

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task297
mkdir -p "$LOG_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

PRODUCTS_DIR=$REPO/products/task297/hrqvae_issue25_phase_ab_joint
mkdir -p "$PRODUCTS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage1_phaseB_${TS}.log"

# 起始 ckpt: task287 Arm A Phase A 100 epoch (κ frozen=0)
INIT_CKPT=$REPO/products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth

echo "[$(date)] Launching Task #297 / Issue #25 Gate 1 — Phase B 30 epoch warm-start"
echo "[$(date)] init_from=$INIT_CKPT"
echo "[$(date)] SAVE_DIR=$PRODUCTS_DIR"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task297_phaseB
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=1e-5 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=100 \
    --lr_theta_post_unfreeze=1e-5 \
    --utilization_freeze_threshold=1.0 \
    --epochs=130 \
    --seed=42 \
    --num_emb_list 128 128 256 \
    --batch_size=256 \
    --ckpt_dir="$PRODUCTS_DIR" \
    --kappa_log_path="$PRODUCTS_DIR/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR/phase_a_baseline.json" \
    --init_encoder_from "$INIT_CKPT" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task297/_TRAINING_PID
echo "[$(date)] Phase B PID=$TRAIN_PID, ckpt_dir=$PRODUCTS_DIR, GPU=1"

echo ""
echo "===== Task #297 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, Phase B 30 epoch warm-start from task287 Arm A Phase A)"
echo "Stage 2/3/4 waiter 需在 Gate 1 PASS 后启动 (scripts/task297_issue25_gate234_waiter.sh)"