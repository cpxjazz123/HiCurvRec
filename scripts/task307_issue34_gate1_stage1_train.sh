#!/bin/bash
# Task #307 / Issue #34 / D9 — Gate 1 Stage 1 100 epoch training launcher
# (per-layer 异构 hash 函数族 on #30 GO 配置 r_l=[0.1,1,10] + s_l=[2,2,2])
#
# 2026-07-30
#
# GATE_DECLARATION (Issue #34 body §Gate 1):
#   gate_1_issue_34: L0/L1/L2 util ≥ 90% at any eval step ≥ ep50
#                    + collision_rate ≤ 0.25
#                    + norm 健康区 ‖x‖_E ∈ [0.7, 0.95] (H3 反证硬停止)
#                    + hash candidates 有效性 (L0 top-3 ≥ 3 unique / L1 top-5 ≥ 5 unique / L2 top-7 ≥ 7 unique)
#                    + loss finite (no NaN/Inf)
#   hard_stop: 任一不满足 → STOP, 不进入 Gate 2

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task307
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task307/hrqvae_issue34_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #307 / Issue #34 / D9 Gate 1: Stage 1 100 epoch"
echo "  --hash_top_k [3,5,7] hardcoded (Issue #34 §3)"
echo "  r_l=[0.1,1,10] + s_l=[2,2,2] (Issue #30 GO 配置)"
echo "  GPU: 0 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task307_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task307_issue34_gate1_stage1_train.py \
    --lr 1e-3 \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path $REPO/HG-Rec/dataset/Instruments/item_emb.parquet \
    --weight_decay 0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task307/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=0"
echo ""
echo "===== Task #307 / Issue #34 / D9 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, per-layer hash wrapper + #30 r_l/s_l GO config, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.25 + norm 健康区 + hash candidates 有效性"
