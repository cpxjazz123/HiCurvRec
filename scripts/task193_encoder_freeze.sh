#!/bin/bash
# Task #193 — Encoder freeze 补充实验 (因果验证)
# 目的: 跑到 epoch 60 (collision 最低点) 后冻结 encoder, 看 collision 劣化是否停止.
# β=0.5 (同 Task #192 B 臂 baseline), 150 epoch 总长 (60 normal + 90 frozen).
#
# 修改 HRQVAE training loop (已 patch 完): epoch_idx == freeze_encoder_epoch 时
#   for p in model.encoder.parameters(): p.requires_grad = False
#
# 使用时机: Task #192 β scan 完成 → 1 GPU 释放 → fire 本脚本.

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
mkdir -p $REPO/logs/task193 $REPO/products/task193

# 同 task192 B 臂 (β=0.5) 一切参数, 只 epochs=150 + freeze_encoder_epoch=60
GPU=${GPU:-0}
OUT_DIR=$REPO/products/task193/hrqvae_beta0.5_freeze60
CACHE_DIR=/home/wlia0047/.triton/cache_task193_freeze
LOG_FILE=$REPO/logs/task193/freeze_arm.log
mkdir -p $OUT_DIR $CACHE_DIR

echo "[$(date)] Start β=0.5 freeze@60 on cuda:$GPU"
cd $REPO/HG-Rec
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=$CACHE_DIR
mkdir -p $TRITON_CACHE_DIR

python3 -u train_hrqvae.py \
    --lr 1e-3 \
    --epochs 150 \
    --batch_size 256 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --weight_decay 0.0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --layers 512 256 128 64 \
    --save_limit 50 \
    --freeze_encoder_epoch 60 \
    --device cuda:0 \
    --ckpt_dir $OUT_DIR \
    > $LOG_FILE 2>&1

echo "[$(date)] DONE freeze (exit=$?); ckpts in $OUT_DIR"
