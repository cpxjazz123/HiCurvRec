#!/bin/bash
# Task #349 / Issue #62 Gate 1 — Stage 1 Arm D 训练 launcher
# (#30 + #43 + K0=256 三联合, 1000 epoch batch=1024 lr=1e-3 sk_eps=0.0)
# Issue #62 owner comment 2026-07-30T16:44:54Z spec
# GPU 0 (R7 空闲, 0% util)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task349
mkdir -p "$LOG_DIR"

export PYTHONPATH=/home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages:$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task349/hrqvae_issue62_gate1_armd
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_armd_${TS}.log

echo "[$(date)] Launching Task #349 / Issue #62 Gate 1 Arm D Stage 1 training" | tee -a "$LOG_FILE"
echo "  Issue #62 Arm D: #30 + #43 + K0=256 三联合" | tee -a "$LOG_FILE"
echo "  --num_emb_list 256 128 256 (K0=256)" | tee -a "$LOG_FILE"
echo "  --num_epochs 1000 --batch_size 1024 --lr 1e-3 --sk_eps 0.0" | tee -a "$LOG_FILE"
echo "  --hyp_c 0.74 (Issue #43 HypPreEncoder)" | tee -a "$LOG_FILE"
echo "  --radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0 (Issue #30)" | tee -a "$LOG_FILE"
echo "  GPU: 0 (R7 空闲)" | tee -a "$LOG_FILE"
echo "  save_path: $SAVE_PATH" | tee -a "$LOG_FILE"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task349
mkdir -p $TRITON_CACHE_DIR
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR

GPU=0
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR" | tee -a "$LOG_FILE"
echo "[$(date)] Training start..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python \
  $REPO/scripts/task349_issue62_gate1_armd_stage1_train.py \
  --ckpt_dir "$SAVE_PATH" \
  --num_emb_list 256 128 256 \
  --num_epochs 1000 \
  --batch_size 1024 \
  --lr 1e-3 \
  --sk_epsilons 0.0 0.0 0.0 \
  --hyp_c 0.74 \
  --use_hyp_pre_encoder True \
  --radius_list 0.1 1.0 10.0 \
  --scale_list 2.0 2.0 2.0 \
  --beta 0.25 \
  --save_limit 1 \
  --seed 42 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Task #349 / Issue #62 Arm D Stage 1 training completed" | tee -a "$LOG_FILE"