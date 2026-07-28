#!/bin/bash
# Task #221 — 逃法二 Gromov Product Stage 1 训练
# 配置: --assignment_mode gromov (argmax Gromov product, 攻前提 a)
# 其余对齐官方: product_manifold + 36d radial + 4d angular, β=0.5
# epochs=200 (Phase 1 探索)
# GPU 1 (R7: 4×L40S 全空闲)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task221/hrqvae_gromov $REPO/logs/task221

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=1
SAVE_DIR=$REPO/products/task221/hrqvae_gromov
LOG_FILE=$REPO/logs/task221/gromov_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #221: assignment_mode=gromov, GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task221_gromov \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 200 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode gromov \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task221/_TRAINING_PID
echo "[$(date)] Task #221 launched PID=$TRAIN_PID" | tee -a $LOG_FILE