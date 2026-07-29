#!/bin/bash
# Task #302 / Issue #31 — Gate 1 Stage 1 launcher (per-layer encoder regularization)
# 2026-07-30
#
# 背景: Issue #31 Gate 0 PASS (scripts/task302_issue31_gate0_wrapper.py + reg test)
#       Gate 1 = Stage 1 100 epoch 端到端训练 per-layer β_l + α_l + γ_l 异构
#       单一变量 vs HG-Rec baseline (R@10=0.1020)
#       R11.3 决策: 直接端到端训练 (绕开 task297 warm-start bug)
#
# GATE_DECLARATION:
#   gate_1_issue_31: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 + ‖x‖_E ≥ 0.3 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 2 空闲 (4×L40S, 当前 GPU 0/1 被 Issue #29/#30 Gate 3 占用 86/87%, GPU 2/3 空闲)
# R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth
# R14: Issue #31 hard-stop at Gate 1 fail

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task302
mkdir -p "$LOG_DIR"

# 使用 genrec_env (Issue #29/#30 已用, 验证 torch 2.11.0+cu130 OK)
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

SAVE_PATH=$REPO/products/task302/hrqvae_issue31_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_${TS}.log

echo "[$(date)] Launching Task #302 / Issue #31 Gate 1: Stage 1 100 epoch per-layer encoder reg"
echo "  --beta_list 0.1 0.3 0.5 (per-layer commit loss 权重)"
echo "  --alpha_list 0.01 0.005 0.001 (per-layer codebook anchor)"
echo "  --gamma_list 0.001 0.0005 0.0001 (per-layer encoder L2)"
echo "  num_emb_list=[64, 128, 256] (baseline K_l)"
echo "  GPU: 2 (R7 空闲, Issue #29/#30 Gate 3 占用 GPU 0/1)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task302_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task302_issue31_gate1_stage1_train.py \
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
    --beta_list 0.1 0.3 0.5 \
    --alpha_list 0.01 0.005 0.001 \
    --gamma_list 0.001 0.0005 0.0001 \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task302/_TRAINING_PID_GATE1
echo "[$(date)] Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=2"
echo ""
echo "===== Task #302 / Issue #31 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 2, per-layer β_l=[0.1,0.3,0.5] + α_l + γ_l, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1h (跟 Issue #29 baseline 同量级)"
echo "Gate 1 hard-stop: 任一不满足 (a-e) → STOP + 关闭 Issue #31"