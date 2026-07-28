#!/bin/bash
# Task #200 Phase 1 v4 — 用户 2026-07-26 5 点清单全套 patch 后的 50 epoch 冒烟
#
# 用户 5 点清单 patch:
#   1. rec_align = ((x_q - z.detach())**2).sum(-1).mean()  [非 F.mse_loss 默认 mean]
#      → emb_rec 梯度从 1e-5 回到 1e-3 量级
#      → patch: HG-Rec/model/utils.py line 652-653
#   2. C' 测试脚本: scripts/task200_diagnose_init_recovery.py (单独跑)
#   3. init_emb 加 print 四项统计 [清单 3 ep1 验证]
#      → patch: HG-Rec/model/utils.py line 339-352 (init_emb 末尾)
#   4. α_geo = 1.0 (上版 v3 已 patch)
#   5. centering_layers = "0,1,2" (上版 v3 已 patch)
#
# Phase 1 v3 FAIL (上版): collision 97% + train_loss 稳定
# Phase 1 v4 (本版): rec_align 改 sum 版 → 期望 collision 改善 + emb_rec 正常更新

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task200 $REPO/products/task200

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=1
SAVE_DIR=$REPO/products/task200/dual_arm_C_v4
LOG_FILE=$REPO/logs/task200/phase1_v4_arm_C.log
rm -f $LOG_FILE

echo "[$(date)] === Phase 1 v4 臂 C 冒烟: dual_codebook + centering 0,1,2 + α_geo=1.0 + rec_align sum 版, 50 epoch, GPU=$GPU ===" | tee -a $LOG_FILE
echo "[$(date)] === 用户清单 1+3 patch 已应用: rec_align sum + init_emb 四项统计 print ===" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task200_v4 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --dual_codebook --centering_layers 0,1,2 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task200/_TRAINING_PID_v4
echo "[$(date)] Phase 1 v4 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

echo "[$(date)] === 监控: ep1 init_emb 四项统计 + collision 改善 (期望 ≤ 50%) + train_loss 稳定 ===" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE