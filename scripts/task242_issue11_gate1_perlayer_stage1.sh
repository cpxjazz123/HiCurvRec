#!/bin/bash
# Task #242 — Issue #11 Gate 1 Arm A: per-layer c_k range Stage 1 40-epoch 训练
# 配置: Task #222 + c_k_range_list=[(1,5), (0.5,20), (0.5,20)] (Issue #11 推荐 Arm A)
#   L0 → U(1, 5)   (期望 82.68% ± 0.29% 一致率)
#   L1 → U(0.5, 20) (期望 67.15% ± 0.22% 一致率)
#   L2 → U(0.5, 20) (期望 75.69% ± 0.79% 一致率)
# Gate 1 通过条件 (Issue #11 §阶段闸门):
#   1. L0 utilization ≥ 90% (项目 §6.7.4 stop-loss (i))
#   2. collision_rate ≤ 0.3706 (task222 best known, task236 统一口径)
# 任一失败 → Gate 1b (Arm A+ 死码字复活), 都失败 → FULL NO-GO 关闭方向
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task242/hrqvae_perlayer_ck $REPO/logs/task242

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task242/hrqvae_perlayer_ck
LOG_FILE=$REPO/logs/task242/perlayer_ck_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #242: Issue #11 Gate 1 Arm A, c_k_range_list=[(1,5),(0.5,20),(0.5,20)], GPU=$GPU ===" | tee $LOG_FILE

# R12: 每个训练必须强制保存 ckpt + 删除旧 ckpt (train_hrqvae.py --save_limit 5 即可)
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task242_perlayer_ck \
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
    --c_k_min 0.5 --c_k_max 20.0 --c_k_seed 42 \
    --c_k_range_list "1:5,0.5:20,0.5:20" \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task242/_TRAINING_PID
echo "[$(date)] Task #242 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
