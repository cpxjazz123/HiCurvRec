#!/bin/bash
# Task #209 Phase 1 臂 A2 — A1 + --scale_norm poincare (commit/code 损失归一化)
#
# 在 A1 (--radii 1.0/1.35/1.70) 基础上, 启用 --scale_norm poincare (Task #203).
# 让 commit/code 损失 = d² / (sinh(√c·ρ)/√c)² — 即 c 只能通过几何形状影响 loss.
# 这与 #84 baseline 损失 (直接 d²) 不同.
# 不启用 --w_path (A3/A4 才有).
#
# 配置: 同 A1, GPU 1
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task209 $REPO/products/task209

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=1
SAVE_DIR=$REPO/products/task209/phase1_arm_A2
LOG_FILE=$REPO/logs/task209/phase1_arm_A2.log
rm -f $LOG_FILE

echo "[$(date)] === Task #209 Phase 1 臂 A2: --radii + --scale_norm poincare, GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task209_arm_A2 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --norm_target 1.0 1.35 1.70 \
    --gamma_norm 0.1 \
    --scale_norm poincare \
    --w_path 0.0 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task209/_TRAINING_PID_arm_A2
echo "[$(date)] Arm A2 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE
