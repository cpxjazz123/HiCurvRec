#!/bin/bash
# Task #152 — L1 对照实验 (用户 2026-07-24 提议)
# 2026-07-24
#
# 目的: 验证 Task #149 L1 κ=0 是 "数据真实偏好" 还是 "θ=0 梯度死区 bug"
# 设计: 改 L1 θ_init=0 → θ_init=+0.15, 其他层 (L0=-0.3, L2=+0.3) 保持 Task #149 配置
#      (L1=0.15 → κ_L1 = 0.5 * tanh(0.15) = 0.0744, 远离 0)
#
# 三种可能结果:
#   (a) L1 漂回 ~0 → "0 是 L1 真实偏好" 硬证据
#   (b) L1 停在 +0.15 附近 → "θ=0 是梯度死区, 是 bug"
#   (c) L1 移动到中间值 → 需要更多分析
#
# 训练配置 (跟 Task #149 main_heterokappa 一致):
#   - Phase A: 100 epoch κ 冻结 (codebook 单独训练)
#   - Phase B: 100 epoch κ 解锁 (lr_theta=1e-5)
#   - total 200 epoch
#   - batch_size=512, lr=1e-3
#   - codebook_size=[64, 128, 256], M=1
#
# R7 GPU: GPU 2 (Task #149 Stage 3 占 GPU 1, Task #151 P5-CID 占 GPU 0)
# 预计: 200 epoch * ~0.4s/ep = ~80s 训练 + 几 min Stage 2 SID

set -eo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task152

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task152
mkdir -p "$LOG_DIR"

CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task152/train/l1_init_0.15
mkdir -p "$CKPT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/l1_control_${TS}.log"

cd /home/wlia0047/ar57/wenyu/GeneRec

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

echo "===== [Task #152] L1 对照实验 launched at $(date) ====="
echo "θ_init_list = [-0.3, +0.15, +0.3]  (vs Task #149 [-0.3, 0, +0.3])"
echo "GPU: 2 (R7 空闲)"
echo "Ckpt dir: $CKPT_DIR"
echo "Log: $LOG_FILE"

# Run with theta_init_list = [-0.3, +0.15, +0.3]
# 关键: L1 从 +0.15 (κ=0.074) 起步, 验证是否漂回 0
nohup python3 scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --kappa_max 0.5 \
    --num_emb_list 64 128 256 \
    --data /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir "$CKPT_DIR" \
    --epochs 200 \
    --batch_size 512 \
    --lr 1e-3 \
    --lr_theta 1e-4 \
    --lr_theta_post_unfreeze 1e-5 \
    --kappa_freeze_epochs 100 \
    --utilization_freeze_threshold 0.05 \
    --theta_init_list -0.3 0.15 0.3 \
    --device cuda:0 \
    --seed 42 \
    --log_interval 10 \
    --kappa_log_path "$CKPT_DIR/kappa_history.json" \
    --dead_code_reset_threshold 2.0 \
    --dead_code_replace_ratio 0.01 \
    --dead_code_reset_every 100 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task152/_TRAINING_PID
echo "[$(date)] task152 training PID: $TRAIN_PID"
echo "log: $LOG_FILE"
echo "theta_init_list = [-0.3, +0.15, +0.3]"