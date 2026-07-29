#!/usr/bin/env bash
# Task #299 / Issue #28 Gate 1 — Stage 1 100 epoch 训练 (per-layer 异构 soft-assign)
#
# 配置 (per Issue #28 body):
# - num_emb_list = [64, 128, 256] (baseline)
# - e_dim = 32 (baseline)
# - per-layer c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)
# - per-layer gumbel_tau_l = [1.0, 0.5, 0.1] (L0=1.0 高温, L1=0.5 中, L2=0.1 低温)
# - 端到端 100 epoch (绕开 task297 K1 warm-start bug)
# - κ not frozen (本 issue 不重复 task144 κ-decouple)
#
# Gate 1 通过条件:
# (a) L0 util ≥ 90% at epoch ≥ 50
# (b) L1 util ≥ 90% at epoch ≥ 50
# (c) L2 util ≥ 90% at epoch ≥ 50
# (d) collision_rate ≤ 0.20
#
# 硬停止: 任一不满足 → STOP, 不进入 Gate 2.

set -euo pipefail

REPO_ROOT="/home/wlia0047/ar57/wenyu/GeneRec"
HG_REC_DIR="$REPO_ROOT/HG-Rec"
DATA_PATH="$HG_REC_DIR/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR="$REPO_ROOT/products/task299"
LOG_DIR="$REPO_ROOT/logs/task299"
mkdir -p "$PRODUCT_DIR/hrqvae_gate1" "$LOG_DIR"

echo "=== Task #299 / Issue #28 Gate 1 启动 ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -1)"
echo ""
echo "Key params:"
echo "  num_emb_list = 64 128 256"
echo "  e_dim = 32"
echo "  c_k_range_list = 1:5,0.5:20,0.5:20"
echo "  gumbel_tau_l = 1.0,0.5,0.1"
echo "  epochs = 100"
echo "  batch_size = 256"
echo "  lr = 1e-3"
echo "  loss_type = poincare"
echo "  dual_codebook --use_centering_list 1 0 0"
echo "  sk_epsilons = 0 0 0 (no Sinkhorn during training)"
echo ""

cd "$HG_REC_DIR"
TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/stage1_gate1_${TS}.log"

# R12: 设置专属 TRITON_CACHE_DIR 防止 sm_89 kernel 加载失败
export TRITON_CACHE_DIR="/home/wlia0047/.triton/cache_task299_gate1"
mkdir -p "$TRITON_CACHE_DIR"

/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3 train_hrqvae.py \
  --data_path "$DATA_PATH" \
  --num_emb_list 64 128 256 \
  --e_dim 32 \
  --layers 512 256 128 \
  --lr 1e-3 \
  --batch_size 256 \
  --epochs 100 \
  --warmup_epochs 20 \
  --lr_scheduler_type linear \
  --loss_type poincare \
  --beta 0.5 \
  --curvatures "1,1,1" \
  --sk_epsilons 0 0 0 \
  --sk_iters 1 \
  --kmeans_init True \
  --kmeans_iters 100 \
  --c_k_range_list "1:5,0.5:20,0.5:20" \
  --gumbel_tau_l "1.0,0.5,0.1" \
  --dual_codebook \
  --centering_layers "0" \
  --kappa_mode fixed \
  --ckpt_dir "$PRODUCT_DIR/hrqvae_gate1" \
  --save_limit 1 \
  --device cuda:0 \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Gate 1 训练完成 ==="
echo "Log: $LOG_FILE"
echo "Best ckpt: $PRODUCT_DIR/hrqvae_gate1/*/best_loss_model.pth"
