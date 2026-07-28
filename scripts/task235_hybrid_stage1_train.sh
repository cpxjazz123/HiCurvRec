#!/bin/bash
# Task #235 — Issue #9 Gate 1: Hybrid per-layer assignment Stage 1 训练
# 配置: --assignment_mode_list per_codeword_kappa,gromov,gromov
#   L0 = Per-Codeword κ c_k ~ U(1,5)  (Issue #9 §Hybrid rule)
#   L1/L2 = Gromov product + argmax   (Task #221 baseline 配置)
# 其余对齐 task220/221: product_manifold + 36d radial + 4d angular, β=0.5
# epochs=40 (early_stop 一致 task222, Gate 1 即可判定)
# GPU 0 (R7: 4×L40S 全空闲)
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task235/hrqvae_hybrid $REPO/logs/task235

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task235/hrqvae_hybrid
LOG_FILE=$REPO/logs/task235/hybrid_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #235 (Issue #9 Gate 1): hybrid assignment_mode_list=pc_k,gromov,gromov, GPU=$GPU ===" | tee $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task235_hybrid \
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
    --assignment_mode_list per_codeword_kappa,gromov,gromov \
    --c_k_min 1.0 --c_k_max 5.0 --c_k_seed 42 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task235/_TRAINING_PID
echo "[$(date)] Task #235 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
