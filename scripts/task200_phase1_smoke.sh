#!/bin/bash
# Task #200 Phase 1 — 双码本 冒烟测试 (臂 C, 50 epoch, GPU 2)
# 用户 2026-07-26 设计, Phase 0 已 PASS.
# 修过上游 (R11.4 critical decision patch):
#   HG-Rec/model/utils.py — HVectorQuantization 加 dual_codebook + use_centering + z_mean EMA
#   HG-Rec/model/utils.py — init_emb 走 emb_geo (球面 kmeans) + emb_rec (latent 均值)
#   HG-Rec/model/utils.py — forward 走双码本 (poincare_distance² commit/code)
#   HG-Rec/model/hrqvae.py — HRQVAE + dual_codebook + use_centering_list
#   HG-Rec/train_hrqvae.py — argparse 加 --dual_codebook + --centering_layers
#   HG-Rec/model/hrqvae_trainer.py — ckpt_dir 加 _dual 后缀
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
SAVE_DIR=$REPO/products/task200/dual_arm_C_smoke
LOG_FILE=$REPO/logs/task200/phase1_arm_C.log
rm -f $LOG_FILE

echo "[$(date)] === Phase 1 臂 C 冒烟: dual_codebook + L0 centering, 50 epoch, GPU=$GPU ===" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task200 \
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
echo "[$(date)] Phase 1 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

echo "[$(date)] === 等到 epoch 50 自动停 (loss 应平稳下降, 无 NaN) ===" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE
echo "[$(date)] Phase 1 后台运行中 (T5-mini 50 epoch ~3-5 min)" | tee -a $LOG_FILE