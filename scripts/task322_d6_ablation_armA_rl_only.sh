#!/bin/bash
# Task #322 / D6 Issue #30 ablation Arm A — r_l only (s_l=[1,1,1])
# Stage 1 100 epoch with per-layer r_l=[0.1, 1.0, 10.0] but s_l baseline [1, 1, 1]
# R12: save_limit=1 + each epoch save best_loss_model.pth
# R7: GPU 1 (R10 — task321 description done, run after task320 completes)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task322
mkdir -p "$LOG_DIR"

export PYTHONPATH=/tmp/genrec_env:$REPO/HG-Rec:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task322/stage1_armA_rl_only
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_armA_${TS}.log

echo "[$(date)] Launching Task #322 Arm A (r_l only)"
echo "  --radius_list 0.1 1.0 10.0 (per-layer 半径缩放 ONLY)"
echo "  --scale_list 1.0 1.0 1.0 (baseline, no s_l lever)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (same as Issue #30)"
echo "  GPU: 1 (R7 等待 task320 释放)"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task322_armA
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
    --scale_list 1.0 1.0 1.0 \
    --c_k_range_list "1.0:5.0,0.5:20.0,0.5:20.0" \
    --c_k_seed 42 \
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task322/_TRAINING_PID_ARMA
echo "[$(date)] Arm A Stage 1 PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #322 / D6 Arm A Launched ====="
echo "PID=$TRAIN_PID (GPU 1, r_l=[0.1,1,10] + s_l=[1,1,1], 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~1-2h"
echo "Gate 1 通过条件: L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50"
echo ""
echo "*** TASK #322 D6 ABLATION 3-ARM PROTOCOL ***"
echo "After Arm A completes (Gate 1 PASS):"
echo "  1. Run scripts/task322_stage2_inference_armA.sh (Sinkhorn)"
echo "  2. Run scripts/task322_stage3_train_armA.sh (T5-mini 200ep)"
echo "  3. Run scripts/task322_stage4_eval_armA.sh (K=20 R@10)"
echo "Then run Arm B (s_l only, --radius_list 1.0 1.0 1.0 --scale_list 2.0 2.0 2.0)"
echo "Then run Arm C (control, full Issue #30: --radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0)"