#!/bin/bash
# Task #200 Phase 1 v5 — 用户 2026-07-26 根因诊断 4 步清单全套 patch 后的 50 epoch 冒烟
#
# 用户根因诊断 (清单 1+2 patch):
#   - cos_mean=0.98 是随机 encoder + ReLU 正象限窄锥的正常现象, 不是 bug
#   - 中心化 EMA 没生效 (EMA 从 0 开始, init_emb 在第一个 batch 时还没收敛)
#   - 不该在随机 encoder 上做初始化, 应该用训好的 baseline encoder
#
# 用户清单 1: 改 init_emb 中心化 (用 batch 均值直接初始化 z_mean)
#   → patch: HG-Rec/model/utils.py init_emb (dual_codebook 分支)
#
# 用户清单 2: 加 --init_encoder_from 从 baseline 热启动 encoder
#   → patch: HG-Rec/train_hrqvae.py argparse
#   → patch: HG-Rec/model/hrqvae_trainer.py fit() 开头 load encoder weights
#
# 之前 patch (清单 1+3 from v4):
#   - rec_align = ((x_q - z.detach())**2).sum(-1).mean() (sum 版)
#   - α_geo = 1.0
#   - centering_layers = "0,1,2"
#   - init_emb 加 print 四项统计
#
# 启动命令: bash scripts/task200_phase1_v5_smoke.sh

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
SAVE_DIR=$REPO/products/task200/dual_arm_C_v5
LOG_FILE=$REPO/logs/task200/phase1_v5_arm_C.log
rm -f $LOG_FILE

# Baseline encoder ckpt (任务 #181 Phase 0.6 训好的)
BASELINE_CKPT_DIR=$REPO/products/task181/hrqvae_fix_v2
BASELINE_CKPT_PATTERN="$BASELINE_CKPT_DIR/*/best_loss_model.pth"
LATEST_BASELINE=$(ls -t $BASELINE_CKPT_PATTERN 2>/dev/null | head -1)
if [ -z "$LATEST_BASELINE" ]; then
    echo "❌ No baseline ckpt found in $BASELINE_CKPT_PATTERN"
    exit 1
fi

echo "[$(date)] === Phase 1 v5 臂 C 冒烟: EMA batch-mean + 热启动 encoder + rec_align sum + α_geo=1.0 + centering 0,1,2 ===" | tee -a $LOG_FILE
echo "[$(date)] === baseline encoder ckpt: $LATEST_BASELINE ===" | tee -a $LOG_FILE
echo "[$(date)] === 用户清单 3 判据: cos_mean<0.3 + util=1.0 + latent_norm_p50 与隔离测试一致 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task200_v5 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --dual_codebook --centering_layers 0,1,2 \
    --init_encoder_from "$LATEST_BASELINE" \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task200/_TRAINING_PID_v5
echo "[$(date)] Phase 1 v5 launched PID=$TRAIN_PID" | tee -a $LOG_FILE

echo "[$(date)] === 监控: ep1 init_emb 四项统计 (期望 cos_mean<0.3) + collision 改善 (期望 ≤ 50%) ===" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE