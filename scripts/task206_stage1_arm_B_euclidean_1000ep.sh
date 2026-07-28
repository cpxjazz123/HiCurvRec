#!/bin/bash
# Task #206 Stage 1 臂 B — 欧式 baseline (R11.3 方案 ii: 不调超参, 接受 collision 差异)
#
# 目的: 跟 #84 HG-Rec baseline 双曲版对比, 看 geometry 是否有本质作用.
# Arm B = --loss_type mse --euclidean_qloss (L2 recon + L2 argmin + L2 commit/code)
# Arm A = #84 baseline (--loss_type poincare, 不加 --euclidean_qloss)
# 两臂其他完全一致: epochs=1000, beta=0.5, num_emb_list=[64,128,256], e_dim=32,
#                   lr=1e-3, kmeans_iters=1000, curvatures=[1,1,1], sk_epsilons=[0,0,0]
#
# 配置 (跟 #84 baseline 完全对齐, 只换 geometry):
#   --euclidean_qloss → L2 distance 在 VQ argmin / commit loss
#   --loss_type mse   → L2 recon loss (取代 --loss_type poincare)
# GPU 0

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206 $REPO/products/task206

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task206/stage1_arm_B_euclidean
LOG_FILE=$REPO/logs/task206/stage1_arm_B_euclidean.log
rm -f $LOG_FILE

echo "[$(date)] === #206 Arm B 1000ep: --loss_type mse --euclidean_qloss (全欧式), GPU=$GPU ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_arm_B \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type mse --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --euclidean_qloss \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task206/_TRAINING_PID_arm_B_1000ep
echo "[$(date)] Arm B 1000ep launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE