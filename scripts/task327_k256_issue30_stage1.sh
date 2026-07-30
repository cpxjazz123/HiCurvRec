#!/bin/bash
# Task #327 / K=256 anchor + Issue #30 per-layer Codebook Transforms — Stage 1
# 跨方向协同: K=256 anchor (task194) + Issue #30 r_l+s_l (task301) + Stage 4 K=50 amplifier (task307)
# GPU 1 (task326 K=384 NO-GO 释放), 100 epoch 跟 Issue #30 task301 一致

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task327
mkdir -p "$LOG_DIR"

SAVE_PATH=$REPO/products/task327/stage1_k256_issue30
mkdir -p $SAVE_PATH

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_k256_issue30_${TS}.log

echo "[$(date)] Launching Task #327 K=256 + Issue #30 Stage 1"
echo "  --num_emb_list 256 128 256 (K=256 anchor)"
echo "  --radius_list 0.1 1.0 10.0 (Issue #30 per-layer)"
echo "  --scale_list 2.0 2.0 2.0 (Issue #30 per-layer)"
echo "  --c_k_range_list '1.0:5.0,0.5:20.0,0.5:20.0' (Issue #30)"
echo "  GPU: 1"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task327_k256_issue30
mkdir -p "$TRITON_CACHE_DIR"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup "$PYTHON_BIN" $REPO/scripts/task327_k256_issue30_stage1_train.py \
    --lr 1e-3 \
    --epochs 100 \
    --batch_size 512 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path $REPO/HG-Rec/dataset/Instruments/item_emb.parquet \
    --weight_decay 0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 256 128 256 \
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
echo "$TRAIN_PID" > $REPO/products/task327/_TRAINING_PID_STAGE1
echo "[$(date)] K=256 + Issue #30 Stage 1 PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #327 / K=256 + Issue #30 Stage 1 Launched ====="
echo "PID=$TRAIN_PID (GPU 1, K=256 + r_l=[0.1,1,10] + s_l=[2,2,2], 100 epoch)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~30 min"