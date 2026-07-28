#!/bin/bash
# Task #211 Phase 1 — C3 臂 Stage 1 训练
# 配置: C1 + path_reg euc (w_path=1.0, path_geometry=euc, 欧式主导)
# 关键对照 (跟 C2 配对决定论文能否叫 "hyperbolic")
# GPU 3
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task211/hrqvae_C3 $REPO/logs/task211

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=3
SAVE_DIR=$REPO/products/task211/hrqvae_C3
LOG_FILE=$REPO/logs/task211/C3_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #211 Phase 1 C3: C1 + path_reg euc (w=1.0, euc 主导), GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task211_C3 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --norm_target 1.0 1.35 1.70 \
    --gamma_norm 0.1 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --w_path 1.0 --path_geometry euc --rho_targets_path 2.0 2.7 3.4 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task211/_TRAINING_PID_C3
echo "[$(date)] C3 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
