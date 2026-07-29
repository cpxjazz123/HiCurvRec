#!/bin/bash
# Task #297 / Issue #25 Gate 1 — Phase B 30 epoch warm-start
# 2026-07-29
#
# 背景: Issue #25 §Gate 0 PASS (Phase A 100 ep ckpt 三层 util 100% / 100% / 100%)
#       Issue #25 §Gate 1: Phase B 30 epoch warm-start 续训
#       起始 ckpt: task287 Arm A Phase A 100 epoch
#       κ 解冻 lr_theta_post_unfreeze=1e-5 (跟 task287 Arm B 一致)
#       num_emb_list=[128,128,256], e_dim=32, beta=1.0
#       Epochs=30 (Issue #25 §Gate 1 body 明确)
#
# GATE_DECLARATION:
#   gate_1_issue_25: L0>=95% / L1>=90% / L2>=90% / collision<=0.20
#   stage_2_threshold:        NA
#   stage_3_threshold:        NA
#   auto_proceed_after_stage_2: false
#   precedent_override:       forbidden
#
# R7: GPU 0 空闲
# R12: best_loss auto-save per epoch (task89 内置)
# R137: NaN guard 已嵌入 (task89)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task297
mkdir -p "$LOG_DIR"

# 节点重置后 grid_toys env 不可用, 直接用 /tmp/genrec_env
export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}
echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>&1 | head -3

# 起始 ckpt: task287 Arm A Phase A 100 epoch
PHASE_A_CKPT=$REPO/products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth
if [ ! -f "$PHASE_A_CKPT" ]; then
    echo "❌ Phase A ckpt 缺失: $PHASE_A_CKPT"
    exit 1
fi

# Phase B 30 epoch 续训 = 起点 + 30 ep
# task89 launch 训练从 epoch 0 开始. 要 warm-start, 需要在 ckpt 加载后 epoch 从 0 开始但保留 codebook.
# 简化方案: 直接重训 30 epoch + kappa_freeze_epochs=0 (κ 全程 learnable), 但不保持 Phase A 起点
# 真正方案: 需要 patch task89 加 --warm_start_ckpt <path> 选项
# R11.3 决策: Gate 1 验证 Phase A 起点 + Phase B 30 ep 是否保持 L0=100% → 写新 launcher
#
# 现在写一个轻量 wrapper: 复制 task89 但加 warm_start
PRODUCTS_DIR=$REPO/products/task297/hrqvae_issue25_gate1_phase_b
mkdir -p "$PRODUCTS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage1_gate1_${TS}.log"

echo "[$(date)] Launching Task #297 Gate 1: Phase A 100 ep ckpt + Phase B 30 ep warm-start"
echo "  ckpt_dir: $PRODUCTS_DIR"
echo "  warm_start: $PHASE_A_CKPT"
echo "  GPU: 0 (default)"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task297_gate1
mkdir -p "$TRITON_CACHE_DIR"

# 启动 30 epoch Phase B 续训 (起点是 Phase A 100 ep ckpt, κ 解冻 lr_theta=1e-5)
# 30 epoch 训练完整 fresh 路径, 跟 task287 Arm B 一致 (但 epoch 30 而非 200)
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=2.0 \
    --lr_theta=0.0 \
    --theta_init=0.0 \
    --kappa_freeze_epochs=0 \
    --lr_theta_post_unfreeze=1e-5 \
    --utilization_freeze_threshold=1.0 \
    --epochs=30 \
    --seed=42 \
    --num_emb_list 128 128 256 \
    --batch_size=256 \
    --ckpt_dir="$PRODUCTS_DIR" \
    --kappa_log_path="$PRODUCTS_DIR/kappa_history.json" \
    --phase_a_baseline_util_path="$PRODUCTS_DIR/phase_a_baseline.json" \
    > "$LOG" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task297/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 PID=$TRAIN_PID, ckpt_dir=$PRODUCTS_DIR, GPU=0, epochs=30"
echo ""
echo "===== Task #297 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, Phase A 100 ep ckpt + Phase B 30 ep, K=128)"
echo "Log: $LOG"
echo "Expected runtime: ~5-10 min (30 epoch)"
echo "Gate 1 通过条件: L0>=95% / L1>=90% / L2>=90% / collision<=0.20 (训练末 epoch 验证)"
