#!/bin/bash
# Task #211 Phase 1 — C2' 臂 (降 w 重试, 用户 "从 w 大往下扫" 实施)
# C2 原始 w_path=1.0 立即 mode collapse, 改 w_path=0.1
# 完整方法 (双曲几何)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task211/hrqvae_C2p $REPO/logs/task211
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task211/hrqvae_C2p
LOG_FILE=$REPO/logs/task211/C2p_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #211 Phase 1 C2': C1 + path_reg hyp w=0.1 (降 w 重试), GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task211_C2p \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --norm_target 1.0 1.35 1.70 --gamma_norm 0.1 \
    --product_manifold --angular_dim 4 --radial_dim 32 \
    --w_path 0.1 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 \
    --save_limit 5 --device cuda:0 --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task211/_TRAINING_PID_C2p
echo "[$(date)] C2' launched PID=$TRAIN_PID" | tee -a $LOG_FILE
