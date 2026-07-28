#!/bin/bash
# Task #210 Phase B2 — product_manifold + path_reg on HYP Stage 1 RQ-VAE 训练.
# 跟 B1 区别: 加 --w_path 1.0 --path_geometry hyp (路径正则化只作用在 hyp subspace).
# GPU 2 (R7: A3 在 GPU 1, B1 在 GPU 0).
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task210/hrqvae_B2
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task210

export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task210_B2
mkdir -p $TRITON_CACHE_DIR

PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task210/_TRAINING_PID_B2

START_TIME=$(date +%s)
echo "=========================================="
echo "Task #210 Phase B2 启动时间: $(date)"
echo "GPU target: cuda:2"
echo "Config: B1 + w_path=1.0 path_geometry=hyp"
echo "=========================================="

python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 40 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --product_manifold \
    --angular_dim 8 \
    --radial_dim 32 \
    --w_path 1.0 \
    --path_geometry hyp \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task210/hrqvae_B2 \
    --device cuda:2 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task210/B2_stage1_train.out &
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE
echo "B2 PID: $TRAIN_PID (saved to $PID_FILE)"

wait $TRAIN_PID

END_TIME=$(date +%s)
echo "=========================================="
echo "Task #210 Phase B2 结束时间: $(date)"
echo "耗时: $((END_TIME - START_TIME)) 秒"
echo "=========================================="
rm -f $PID_FILE
