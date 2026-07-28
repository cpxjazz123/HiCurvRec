#!/bin/bash
# Task #222 — Phase 2 早停 30 epoch 复现 Task #220 healthy ckpt
# 配置完全同 Task #220, 只改 epochs=40 + GPU=2
# monitor 在 ep30 检查 utilization + collision, 不达标则报告 NO-GO
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task222/hrqvae_pck_replay $REPO/logs/task222

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task222/hrqvae_pck_replay
LOG_FILE=$REPO/logs/task222/pck_replay_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #222: replay Task #220 config, epochs=40, GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task222_pck_replay \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 40 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode per_codeword_kappa \
    --c_k_min 0.5 --c_k_max 5.0 --c_k_seed 42 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task222/_TRAINING_PID
echo "[$(date)] Task #222 launched PID=$TRAIN_PID" | tee -a $LOG_FILE