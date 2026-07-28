#!/bin/bash
# Task #206-rev2 Stage 1 双臂并行 (修复 + 高 save_limit 拿高 collision ckpts)
#
# A. 双曲 baseline (#84-style) 50 epochs, --save_limit 50, --eval_step 1  → 拿 epoch 14/19/24 高 collision ckpt
# B. 欧式 Arm B (F.mse_loss → sum(-1).mean fix) 50 epochs, 同样          → 找 collision 跟 A epoch 14/19 对齐的 epoch
#
# 决策触发:
#   - 双曲 epoch 14 collision ≈ 67.5%
#   - 双曲 epoch 19 collision ≈ 44.4%
#   - 欧式 epoch X 应该落在这两个值之间或附近 (匹配), 跑 Stage 2+3+4
#   - 如果欧式 50 epochs 后 collision 还 > 80%, 需要更多 epochs
#
# GPU: 0 = 双曲 baseline, 2 = 欧式 Arm B FIX (GPU 1 被 Stage 3 T5 占用, 还在跑)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

# ===== 臂 A: 双曲 baseline (跟 #84 完全一致, β=1.0) =====
GPU_A=0
SAVE_DIR_A=$REPO/products/task206/stage1_baseline_retrain
LOG_FILE_A=$REPO/logs/task206/stage1_baseline_retrain.log
rm -f $LOG_FILE_A

echo "[$(date)] === #206-rev2 Step 2A 双曲 baseline 重训 50ep, save_limit=50, eval_step=1, GPU=$GPU_A ===" | tee $LOG_FILE_A
CUDA_VISIBLE_DEVICES=$GPU_A TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206c_baseline \
nohup python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 1.0 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR_A \
    > $LOG_FILE_A 2>&1 &
PID_A=$!
echo "[$(date)] 双曲 baseline 重训 PID=$PID_A" | tee -a $LOG_FILE_A
echo $PID_A > $REPO/products/task206/_TRAINING_PID_baseline_retrain

# ===== 臂 B: 欧式 FIX (--loss_type mse --euclidean_qloss, β=0.5) =====
GPU_B=2
SAVE_DIR_B=$REPO/products/task206/stage1_euclidean_fix
LOG_FILE_B=$REPO/logs/task206/stage1_euclidean_fix.log
rm -f $LOG_FILE_B

# 等 5 秒让 A 先启动, 避免 GPU 同时初始化冲突
sleep 5

echo "[$(date)] === #206-rev2 Step 2B 欧式 FIX 重训 50ep, save_limit=50, eval_step=1, GPU=$GPU_B ===" | tee $LOG_FILE_B
CUDA_VISIBLE_DEVICES=$GPU_B TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206c_euclidean_fix \
nohup python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type mse --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --euclidean_qloss \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR_B \
    > $LOG_FILE_B 2>&1 &
PID_B=$!
echo "[$(date)] 欧式 FIX 重训 PID=$PID_B" | tee -a $LOG_FILE_B
echo $PID_B > $REPO/products/task206/_TRAINING_PID_euclidean_fix

echo "[$(date)] === 两臂并行启动: 双曲 PID=$PID_A (GPU 0) + 欧式 PID=$PID_B (GPU 2) ==="
echo "[$(date)] 监控: tail -f $LOG_FILE_A   双曲 baseline"
echo "[$(date)] 监控: tail -f $LOG_FILE_B   欧式 FIX"