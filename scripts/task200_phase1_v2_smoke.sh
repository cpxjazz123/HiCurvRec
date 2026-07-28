#!/bin/bash
# Task #200 Phase 1 v2 — 双码本 forward 修复方向 D 重跑 (臂 C, 50 epoch, GPU 2)
# 用户 2026-07-26 修复方向 D: commit/code 用欧氏 MSE (而非 poincare_distance).
# emb_geo / emb_rec / latent 全部在切空间, expmap0 只在算 d 和 geo_loss 两行出现.
# Phase 1 v1 FAIL root cause: poincare_distance(emb_rec, z_for_assign) 在 ‖x‖>1 时 lambda_x 变负 → train_loss 飞涨.
#
# 上游 patch:
#   HG-Rec/model/utils.py — HVectorQuantization dual_codebook forward:
#     commit/code 改 F.mse_loss (切空间欧氏距离), expmap0 保留在算 d 一行
# 备份在 *.bak200 (rollback 路径)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task200 $REPO/products/task200

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task200/dual_arm_C_v2
LOG_FILE=$REPO/logs/task200/phase1_v2_arm_C.log
rm -f $LOG_FILE

echo "[$(date)] === Phase 1 v2 臂 C 冒烟: dual_codebook + L0 centering + commit/code 用 MSE, 50 epoch, GPU=$GPU ===" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task200_v2 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --dual_codebook --centering_layers 0 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task200/_TRAINING_PID
echo "[$(date)] Phase 1 v2 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

echo "[$(date)] === 监控指标: train_loss 稳定 / collision ≤ 30% / cos_max < 0.95 ===" | tee -a $LOG_FILE
echo "[$(date)] === 等到 epoch 50 自动停 (T5-mini 50 epoch ~3-5 min) ===" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE