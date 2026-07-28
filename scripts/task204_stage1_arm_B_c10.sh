#!/bin/bash
# Task #204 Stage 1 臂 B — c=10 fixed + quant_loss_weight=1.0 (用户 2026-07-26 决定性实验)
#
# 目的: 等价于 #199 D c=10 (collision=8.26%) 重跑, 验证 #204 A 是否跟它碰撞率相同.
# 配置: c=10 + 默认 weight=1.0. 是 c=10 的"标准" baseline 重做.
#
# 配置 (跟 #199 D c=10 对齐):
#   epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
#   lr=1e-3, kmeans_iters=1000, loss_type=poincare, curvatures=[10,10,10], quant_loss_weight=1.0 (默认)
# GPU 2

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task204 $REPO/products/task204

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task204/stage1_arm_B_c10
LOG_FILE=$REPO/logs/task204/stage1_arm_B.log
rm -f $LOG_FILE

echo "[$(date)] === #204 臂 B: c=[10,10,10] + quant_loss_weight=1.0 (默认), GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task204_arm_B \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 10.0,10.0,10.0 \
    --quant_loss_weight 1.0 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task204/_TRAINING_PID_arm_B
echo "[$(date)] Arm B launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE