#!/bin/bash
# Task #306 / Issue #33 / D8 — Gate 1 Stage 1 100 epoch training launcher
# (per-item 软分配 on #30 GO 配置 r_l=[0.1,1,10] + s_l=[2,2,2])
#
# 2026-07-30
#
# GATE_DECLARATION (Issue #33 body §Gate 1):
#   gate_1_issue_33: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50
#                    + norm 健康区 ‖x‖_E ∈ [0.7, 0.95] (H3 反证硬停止)
#                    + loss finite (no NaN/Inf)
#   hard_stop: 任一不满足 → STOP, 不进入 Gate 2
#   precedent_override: forbidden
#
# R7: GPU 0 空闲 (2026-07-30 04:13 检查, 4 卡全空闲)
# R12: save_limit=1 + 训练结束强制保存 best_ckpt (删旧)
# R14: Issue #33 hard-stop at Gate 1 fail

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task306
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task306/hrqvae_issue33_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #306 / Issue #33 / D8 Gate 1: Stage 1 100 epoch"
echo "  --temperature 1.0 (per-item 软分配)"
echo "  --radius_list 0.1 1.0 10.0 (#30 GO per-layer 半径缩放)"
echo "  --scale_list 2.0 2.0 2.0 (#30 GO per-layer scale factor)"
echo "  GPU: 0 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task306_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task306_issue33_gate1_stage1_train.py \
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
echo "$TRAIN_PID" > $REPO/products/task306/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=0"
echo ""
echo "===== Task #306 / Issue #33 / D8 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 0, per-item soft + #30 r_l/s_l GO config, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 + norm 健康区 ‖x‖_E ∈ [0.7, 0.95]"
