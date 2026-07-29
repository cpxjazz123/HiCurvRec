#!/bin/bash
# Task #304 / D6 ablation Arm B — Gate 1 Stage 1 100 epoch 训练 (s_l only)
# 2026-07-30
#
# 背景: Task #304 D6 ablation Arm B (s_l only) Gate 0 PASS
#       config: r_l=[1,1,1] + s_l=[2,2,2] + c_k_range=[(1,5),(0.5,20),(0.5,20)]
#       跟 Issue #30 (r_l+s_l) 唯一区别: r_l 取消 [0.1,1,10] → [1,1,1] baseline
#       ckpt_dir = products/task304/d6_arm_b_gate1/
#
# GATE_DECLARATION:
#   gate_1_d6_arm_b: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50
#   auto_proceed_after_stage_0:    true
#   precedent_override:            forbidden
#
# R7: GPU 1 空闲
# R12: save_limit=1 + 每个 epoch 强制保存 best_loss_model.pth

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task304
mkdir -p "$LOG_DIR"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

SAVE_PATH=$REPO/products/task304/d6_arm_b_gate1
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_arm_b_gate1_${TS}.log

echo "[$(date)] Launching Task #304 / D6 Arm B: Stage 1 100 epoch (s_l only)"
echo "  --radius_list 1.0 1.0 1.0 (per-layer 半径缩放, baseline 取消 Issue #30 [0.1,1,10])"
echo "  --scale_list 2.0 2.0 2.0 (per-layer scale factor, 跟 Issue #30 相同)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (per-layer c_k range)"
echo "  GPU: 1 (R7 空闲)"
echo "  save_path: $SAVE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task304_arm_b_gate1
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup $PYTHON_BIN $REPO/scripts/task304_d6_gate1_stage1_train_arm_b.py \
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
    --radius_list 1.0 1.0 1.0 \
    --scale_list 2.0 2.0 2.0 \
    --c_k_range_list "1.0:5.0,0.5:20.0,0.5:20.0" \
    --c_k_seed 42 \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task304/_TRAINING_PID_ARM_B_GATE1
echo "[$(date)] Arm B Gate 1 Stage 1 training PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #304 / D6 Arm B Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, s_l only + c_k_range, 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~30-50 min"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50"