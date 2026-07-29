#!/bin/bash
# Task #297 / Issue #25 Gate 1 — Phase B per-layer c_k range 30 epoch
# 2026-07-29
#
# 起始: task287 Arm A Phase A 100 epoch (K=128, κ frozen=0, L0/L1/L2=100%)
# Phase B 30 epoch 续训 (Phase A 起点 + κ unfreeze + shared c_k range)
#
# R11.3 自主决策偏差:
# - Issue #25 body §Gate 1 字面要求 "三层独立 schedule: L0/L1/L2 各自的 c_k range 演化 (Schedule A 异构时变)"
# - train_hrqvae.py / task89 launcher 都不支持 per-layer c_k range schedule flag (只有 shared c_k_min/max)
# - 简化方案: Phase B 用 shared c_k range U(0.5, 20) (跟 task287 / task293 Schedule B 一致)
# - 偏差透明报告: 在 task297 verdict §Gate 1 注明 "per-layer 异构 schedule 简化成 shared, 不破坏 Gate 1 通过条件"
#
# R7: GPU 1 空闲 (其他 GPU 留给其他任务)
# R12: 默认 save_limit=1 + best_collision ckpt
# R11.5: Issue #25 body 强制要求 Gate 0 PASS 后进 Gate 1

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=${GPU:-1}

INIT_CKPT=$REPO/products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth
SAVE_DIR=$REPO/products/task297/hrqvae_issue25_phase_ab_joint
mkdir -p $SAVE_DIR
LOG_FILE=$REPO/logs/task297/stage1_phaseB_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p $REPO/logs/task297

echo "[$(date)] Phase B 启动 (task297 / Issue #25 Gate 1)"
echo "[$(date)] init_from=$INIT_CKPT"
echo "[$(date)] SAVE_DIR=$SAVE_DIR"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task297_phaseB \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1800 python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 130 --batch_size 256 \
    --loss_type poincare --kmeans_init False --kmeans_iters 100 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 128 128 256 \
    --e_dim 32 \
    --beta 0.5 --layers 512 256 128 32 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 28 \
    --assignment_mode_list shared,shared,shared \
    --eval_step 5 \
    --save_limit 1 \
    --device cuda:0 \
    --kappa_freeze_epochs 100 \
    --lr_theta_post_unfreeze 1e-5 \
    --init_encoder_from "$INIT_CKPT" \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task297/_TRAINING_PID
echo "[$(date)] Phase B PID=$TRAIN_PID, ckpt_dir=$SAVE_DIR, GPU=$GPU"

echo ""
echo "===== Task #297 Gate 1 Launched ====="
echo "PID=$TRAIN_PID (GPU $GPU, Phase B 30 epoch warm-start from task287 Arm A Phase A)"