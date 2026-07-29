#!/bin/bash
# Task #315 / Issue #34 simplified D9 多样 hash — Gate 1 Stage 1 100 epoch 训练
# 2026-07-30
#
# 背景: Issue #34 D9 简化实施 (R11.5). Stage 1 = task301 Issue #30 GO 端点 (per-layer r_l + s_l)
#       沿用 task301 geometry: r_l=[0.1,1.0,10.0], s_l=[2.0,2.0,2.0]
#       per-layer hash 函数族在 Stage 1 关闭 (仅在 Stage 2 推断时开启 top-k candidates)
#       ckpt_dir = products/task315/hrqvae_issue34_d9_gate1/
#
# GATE_DECLARATION:
#   gate_1_issue_34: L0/L1/L2 util ≥ 90% + collision ≤ 0.25 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 1 空闲
# R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth
# R14: Issue #34 hard-stop at Gate 1 fail (沿用 #30 Gate 1 通过条件)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task315
mkdir -p "$LOG_DIR"

export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task315/hrqvae_issue34_d9_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #315 / Issue #34 simplified D9 Gate 1: Stage 1 100 epoch"
echo "  --radius_list 0.1 1.0 10.0 (per-layer 半径缩放, 沿用 #30)"
echo "  --scale_list 2.0 2.0 2.0 (per-layer scale factor, 沿用 #30)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (per-layer c_k range, 沿用 #30)"
echo "  GPU: 1 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task315_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup python3 $REPO/scripts/task301_issue30_gate1_stage1_train.py \
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
    --loss_mult_codebook 1.0 \
    --layers 512 256 128 64 \
    --save_limit 1 \
    --ckpt_dir $SAVE_PATH/ \
    --radius_list 0.1 1.0 10.0 \
    --scale_list 2.0 2.0 2.0 \
    --c_k_range_list "1.0:5.0,0.5:20.0,0.5:20.0" \
    --c_k_seed 42 \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task315/_TRAINING_PID_GATE1
echo "[$(date)] Task #315 Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #315 / Issue #34 simplified D9 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, per-layer Codebook Transforms, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h (similar to task301)"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.25 at ep ≥ 50"
