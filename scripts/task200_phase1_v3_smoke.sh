#!/bin/bash
# Task #200 Phase 1 v3 — 用户 2026-07-26 拍板 5 点修正方案冒烟 (臂 C, 50 epoch, GPU 2)
#
# 用户 5 点修正:
#   1. α_geo = 1.0 (不再是 0.1)
#      → patch: HG-Rec/model/utils.py HVectorQuantization.forward
#      → 验证: 隔离测试 v4 emb_geo grad ×10 (0.0024 → 0.0224)
#   2. --centering_layers 默认 "0,1,2" (三层都启用, 不只是 L0)
#      → patch: HG-Rec/train_hrqvae.py argparse default
#      → 机制: L1/L2 残差也含 L0 共同偏移 (量化误差), 必须中心化才能露出个体差异
#   3. emb_rec 保持自由 (未改, F.normalize/mean 是 kmeans 推导而非强制)
#   4. 隔离测试 v4 PASS (A 三层 init PASS / B 三层 forward PASS / C 因 init 满无法观察, 但 grad ×10 确认)
#   5. 停止在流水线打补丁 → 走隔离测试验证 → 通过后才进流水线
#
# Phase 1 v2 FAIL (上版): α_geo=0.1 + centering 0 → collision 90% + train_loss 飞涨 15720
# Phase 1 v3 (本版): α_geo=1.0 + centering 0,1,2 → 预期 collision 改善 + train_loss 稳定
#
# 上游 patch (rollback 路径 *.bak200):
#   HG-Rec/model/utils.py — HVectorQuantization forward: alpha_geo 0.1 → 1.0
#   HG-Rec/train_hrqvae.py — argparse default centering_layers "0" → "0,1,2"

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
SAVE_DIR=$REPO/products/task200/dual_arm_C_v3
LOG_FILE=$REPO/logs/task200/phase1_v3_arm_C.log
rm -f $LOG_FILE

echo "[$(date)] === Phase 1 v3 臂 C 冒烟: dual_codebook + centering 0,1,2 + α_geo=1.0, 50 epoch, GPU=$GPU ===" | tee -a $LOG_FILE
echo "[$(date)] === 用户 5 点修正方案: isolation_test_v4 PASS, emb_geo grad ×10 已验证 ===" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task200_v3 \
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
echo $TRAIN_PID > $REPO/products/task200/_TRAINING_PID_v3
echo "[$(date)] Phase 1 v3 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

echo "[$(date)] === 监控指标: train_loss 稳定 / collision 改善 (<= 50%) / cos_max < 0.95 ===" | tee -a $LOG_FILE
echo "[$(date)] === 等到 epoch 50 自动评估 (T5-mini 50 epoch ~3-5 min) ===" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE