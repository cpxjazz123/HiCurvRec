#!/bin/bash
# Task #206 Stage 1 臂 B (方案 iii) — 欧式 baseline, 2000 epoch 看 natural collision 收敛
#
# 目的: 验证 1000ep Arm B 是否已达 plateau. 如果 2000ep collision 明显下降 →
#       1000ep 是欠训, 需要更多 epoch; 如果 2000ep ≈ 1000ep → 1000ep 已收敛.
# 这是 decision 锚点, 验证 1000ep 读数稳定, 防被"半训"误导.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206 $REPO/products/task206

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task206/stage1_arm_B_euclidean_2000ep
LOG_FILE=$REPO/logs/task206/stage1_arm_B_euclidean_2000ep.log
rm -f $LOG_FILE

echo "[$(date)] === #206 Arm B 2000ep (sanity iii): --loss_type mse --euclidean_qloss, GPU=$GPU ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_arm_B_2000ep \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 2000 --batch_size 256 \
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
echo $TRAIN_PID > $REPO/products/task206/_TRAINING_PID_arm_B_2000ep
echo "[$(date)] Arm B 2000ep launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE