#!/usr/bin/env bash
# Task #305 / Issue #33 — Stage 1 Gate 1 训练 launcher
# Issue #33 Gate 1: 100 epoch 训练 per-layer r_l=[0.1,1,10] + s_l=[2,2,2] + per-item soft-assign τ=1.0

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task305
mkdir -p $TRITON_CACHE_DIR

# Issue #33 端点配置 (沿用 #30 GO: r_l=[0.1,1,10]+s_l=[2,2,2], 加 per-item soft-assign)
python3 scripts/task305_issue33_gate1_stage1_train.py \
  --device cuda:1 \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-3 \
  --radius_list 0.1 1.0 10.0 \
  --scale_list 2.0 2.0 2.0 \
  --per_item_tau_list 1.0 1.0 1.0 \
  --per_item_softassign_enabled 1 \
  --loss_type poincare \
  --beta 0.5 \
  --ckpt_dir ./ckpt/Instruments_issue33_peritem_softassign \
  --save_limit 1
