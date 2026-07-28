#!/bin/bash
# Task #253 (新) — Issue #13 Gate 2: Möbius 残差算子实际 Stage 1 训练
# 改动: utils.py:1795 残差算子替换为 mobius_add(-x_res_hyp, residual_hyp, c)
#       + euc part 保持欧式减法 (product_manifold 兼容)
# 配置跟 task222 ep29 healthy ckpt 一致: product_manifold e_dim=36 hyp=4 euc=32
#   β=0.5, codebook=[64,128,256], sk_epsilons=0 (no-Sinkhorn), kmeans_init
# 短训 50 epoch (Issue #13 §Gate 2 明文)
# GPU 0 (R7: Task #243 占 1/2, 0/3 空闲)
# R12: save_limit=2 (保 latest ckpt only)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task253/hrqvae_mobius_residual $REPO/logs/task253

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task253/hrqvae_mobius_residual
LOG_FILE=$REPO/logs/task253/mobius_residual_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #253 (Issue #13 Gate 2): Möbius 残差算子 Stage 1, GPU=$GPU ===" | tee $LOG_FILE
echo "[$(date)] Patch: utils.py:1795 residual = mobius_add(-x_res_hyp, residual_hyp, c) [hyp] + euc 欧式" | tee -a $LOG_FILE
echo "[$(date)] Stage 1 config: product_manifold e_dim=36 β=0.5 [64,128,256] sk=0 50 epoch" | tee -a $LOG_FILE
echo "[$(date)] Backup: /home/wlia0047/.claude/jobs/04ccf474/tmp/utils.py.issue253.bak" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task253_mobius \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode_list shared,shared,shared \
    --save_limit 2 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task253/_TRAINING_PID
echo "[$(date)] Task #253 launched PID=$TRAIN_PID" | tee -a $LOG_FILE