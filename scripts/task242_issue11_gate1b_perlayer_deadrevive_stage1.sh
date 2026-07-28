#!/bin/bash
# Task #242 — Issue #11 Gate 1b (Arm A+): per-layer c_k_range + dead-codeword revival
# 配置: Task #242 Arm A 配置 + --anti_collapse dead_revive (每 eval step 复活死码字).
# Gate 1b 通过条件 (Issue #11 §阶段闸门): 同 Gate 1 (L0 util ≥ 90% AND collision ≤ 0.3706).
# 任一失败 → FULL NO-GO 关闭方向.
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task242/hrqvae_perlayer_ck_deadrevive $REPO/logs/task242

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task242/hrqvae_perlayer_ck_deadrevive
LOG_FILE=$REPO/logs/task242/perlayer_ck_deadrevive_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #242 Gate 1b: Arm A+ c_k_range_list + dead_revive, GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task242_perlayer_ck_deadrevive \
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
    --anti_collapse dead_revive \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task242/_TRAINING_PID_GATE1B
echo "[$(date)] Task #242 Gate 1b launched PID=$TRAIN_PID" | tee -a $LOG_FILE
