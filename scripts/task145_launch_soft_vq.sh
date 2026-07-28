#!/bin/bash
# Task #145 — 软量化退火 Phase 2 (Soft VQ-VAE 风格)
# 2026-07-24
#
# 验证用户提议: 不依赖 θ (κ) 的梯度, 直接用 softmax-weighted soft assignment
# + τ annealing 解决 free-curv codebook collapse. τ 1.0 → 0.01 over 200 epoch
# (linear), 早期 uniform soft (每 codeword 拿梯度), 末期 near-hard.
#
# 关键设计:
#   1. Straight-through estimator: forward = hard assignment, backward = soft weights
#   2. τ 退火: epoch 0 → 1.0 (uniform), epoch 200 → 0.01 (near argmax)
#   3. codebook + encoder/decoder 走正常 Adam, θ 不参与梯度 (κ 冻结)
#   4. baseline 跟 Task #144 Arm A 一致 (M=1, kappa_max=2.0, θ frozen), 单变量
#
# R7: GPU 3 空闲 (0% util, 0 MiB). 用 GPU 3.
# R12: best_loss auto-save per epoch (已嵌入 task89_stage1_train_rqvae.py).
# R137: NaN guard 已嵌入.

set -uo pipefail
cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task145
mkdir -p "$LOG_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# ===== Task #145: 软量化退火 + κ 冻结 (单臂) =====
PRODUCTS_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task145/train/main_softvq
mkdir -p "$PRODUCTS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/task145_main_${TS}.log"

echo "[$(date)] Launching Task #145: 软量化退火 Phase 2 — log: $LOG"

# Triton cache isolation (per CLAUDE.md GPU table, sm_89 L40S)
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task145_main
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=200 \
    --utilization_freeze_threshold=1.0 \
    --soft_vq_enabled \
    --soft_vq_tau_start=1.0 \
    --soft_vq_tau_end=0.01 \
    --soft_vq_anneal_type=linear \
    --epochs=200 \
    --seed=42 \
    --ckpt_dir="$PRODUCTS_DIR" \
    --kappa_log_path="$PRODUCTS_DIR/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR/phase_a_baseline.json" \
    > "$LOG" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > /home/wlia0047/ar57/wenyu/GeneRec/products/task145/_TRAINING_PID
echo "[$(date)] Task #145 PID=$TRAIN_PID, ckpt_dir=$PRODUCTS_DIR, GPU=3"
echo "[$(date)] GPU 3 status: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 3)"
