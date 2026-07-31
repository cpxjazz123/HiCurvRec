#!/bin/bash
# Task #350 / Issue #62 Gate 1 — Stage 1 Arm C 训练 launcher
# (#30 + #43 联合, K=[64,128,256] baseline + Sinkhorn ON, 1000 epoch batch=1024 lr=1e-3)
# Issue #62 §Gate 1 Arm C spec
# GPU 0 (R7 空闲, 0% util)
set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task350
mkdir -p "$LOG_DIR"

export PYTHONPATH=/home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages:$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

SAVE_PATH=$REPO/products/task350/hrqvae_issue62_gate1_armc
mkdir -p $SAVE_PATH
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage1_gate1_armc_${TS}.log

echo "[$(date)] Launching Task #350 / Issue #62 Gate 1 Arm C Stage 1 training" | tee -a "$LOG_FILE"
echo "  Issue #62 Arm C: #30 + #43 联合 (baseline K + Sinkhorn ON)" | tee -a "$LOG_FILE"
echo "  --num_emb_list 64 128 256 (Issue #30 baseline K, NO K0=256)" | tee -a "$LOG_FILE"
echo "  --num_epochs 1000 --batch_size 1024 --lr 1e-3 --sk_eps 0.003 (Sinkhorn ON)" | tee -a "$LOG_FILE"
echo "  --hyp_c 0.74 (Issue #43 HypPreEncoder)" | tee -a "$LOG_FILE"
echo "  --radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0 (Issue #30)" | tee -a "$LOG_FILE"
echo "  GPU: 0 (R7 空闲)" | tee -a "$LOG_FILE"
echo "  save_path: $SAVE_PATH" | tee -a "$LOG_FILE"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task350
mkdir -p $TRITON_CACHE_DIR
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR

GPU=0
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR" | tee -a "$LOG_FILE"
echo "[$(date)] Training start..." | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python \
  $REPO/scripts/task350_issue62_gate1_armc_stage1_train.py \
  --ckpt_dir "$SAVE_PATH" \
  --num_emb_list 64 128 256 \
  --num_epochs 1000 \
  --batch_size 1024 \
  --lr 1e-3 \
  --sk_epsilons 0.003 0.003 0.003 \
  --hyp_c 0.74 \
  --use_hyp_pre_encoder True \
  --radius_list 0.1 1.0 10.0 \
  --scale_list 2.0 2.0 2.0 \
  --beta 0.25 \
  --save_limit 1 \
  --seed 42 2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] Task #350 / Issue #62 Arm C Stage 1 training completed" | tee -a "$LOG_FILE"