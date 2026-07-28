#!/bin/bash
# Task #163 Phase D — paper-faithful HG-Rec free-curv with ORC-based θ_init
# 用户 2026-07-24 23:21 提议: 用 ORC 子图 avg_clustering 测量每层 θ_init
# 关键差异 vs Task #162: θ_init 从 random [0.084, -0.285, -0.135] → ORC-derived [-0.3161, -0.7997, -0.7997]
# 期望: ORC-based init 是 hyperbolic (-1.3) vs β=0.5 训练终点 spherical (+1.1) — 看 init 信号能否影响 κ 终点
# 复用 Task #162 paper-faithful 配置: β=0.5, [32,64,256], sk=0, B 方案关闭
# 输出: products/task163/rqvae_orc_init/<ts>/best_ckpt.pth + κ_history.json

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/orc_stage1_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task163/rqvae_orc_init/${TS}
mkdir -p $LOG_DIR $PROD_DIR

PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #163 Phase D] ORC-based θ_init paper-faithful HG-Rec free-curv launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: paper-faithful β=0.5 + [32,64,256] + sk=0 + ORC-derived θ_init [-0.3161, -0.7997, -0.7997]" | tee -a $LOG_FILE
echo "GPU 2 (CUDA_VISIBLE_DEVICES=2, completely idle)" | tee -a $LOG_FILE
echo "θ_init per-layer (ORC-derived from 32 L0 groups + 296 L0×L1 prefix groups):" | tee -a $LOG_FILE
echo "  L0 = -0.3161 (L0 avg_cc mean=0.342, shifted=-0.158, hyperbolic)" | tee -a $LOG_FILE
echo "  L1 = -0.7997 (L0×L1 prefix avg_cc mean=0.100, shifted=-0.400, hyperbolic, hit cap)" | tee -a $LOG_FILE
echo "  L2 = -0.7997 (L2 skipped per user risk #1, use L1 placeholder)" | tee -a $LOG_FILE
echo "  → κ_init = [-0.6119, -1.3277, -1.3277] (κ_max=2.0, θ small negative → negative κ)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs Task #162: random init κ=[+0.167, -0.555, -0.268] → 训练终值 κ=[+1.146, +0.655, +0.745] spherical" | tee -a $LOG_FILE
echo "vs Task #163 (本次): ORC init κ=[-0.612, -1.328, -1.328] → 期望 κ 跟 #162 不同 (具体方向待观察)" | tee -a $LOG_FILE
echo "Stage 1 fork: scripts/task89_stage1_train_rqvae.py" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=2 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 1000 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 5e-3 \
    --theta_init_list -0.3161 -0.7997 -0.7997 \
    --kappa_max 2.0 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --dead_code_reset_every 0 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --log_interval 20 \
    --save_every 100 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #163 Phase D] ORC-init paper-faithful HG-Rec free-curv completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE
