#!/bin/bash
# Task #137 — Task #89 Stage 1 retrain (R137 fix applied)
# R137 fix: hrqvae_free_curv.py 4 hard branches → torch.where unified operator
# 顺序跑 3 臂 (A=M=1, B=M=2, C=M=3) × 1000 epoch, 各 2 小时
# GPU 1 空闲 (R7 — TIGER GPU 0, ETEGRec GPU 2)
# 输出到 products/task137/ (与 task89 隔离, 保留 bug-polluted baseline)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task137
mkdir -p $LOG_DIR $PROD_DIR/{train/arm_A_M1,train/arm_B_M2,train/arm_C_M3}

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task137_free_curv_retrain_${TS}.log
PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #137] launch at $(date) =====" | tee $LOG_FILE
echo "R137 fix applied: hrqvae_free_curv.py 4 hard branches → torch.where" | tee -a $LOG_FILE
echo "GPU 1 (R7 — TIGER GPU 0 busy, ETEGRec GPU 2 busy)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "R137 critical: θ_init=0.01 (NOT 0!) to escape Euclidean fixed point" | tee -a $LOG_FILE
echo "  θ=0 → κ=0 → eucl branch → no κ gradient → math fixed point (R137 verified)" | tee -a $LOG_FILE
echo "  θ=0.01 → κ=0.02 → sph branch → grad flows → κ can learn (R137 verified probe)" | tee -a $LOG_FILE

# R11.3 自决: seed=42 与 Task #89 baseline 一致 (单 seed 验证 per [[user-no-multiseed-override]])
# R11.3 自决: θ_init=0.01 (NOT 0!) — R137 验证: θ=0 是数学 fixed point (eucl 分支不依赖 κ)
#                          θ=0.01 进入 sph 分支, 让 κ_m 能学习, 才能验证 "数据本质欧氏" 结论
# R11.3 自决: κ_max=0.5 (vs 原 spec 2.0) — R137 训练 ep 200→260 出现 NaN (κ→κ_max 边界 acos 不稳定),
#                          降 κ_max 到 0.5 (跟 HG-Rec c555 baseline 一致) + lr_theta 1e-3 (普通 lr) 数值稳定.
# R11.3 自决: lr_theta=1e-3 (vs 原 spec 5e-3) — κ_max 减小后, lr_theta 也按比例减小, 避免饱和区梯度爆炸

for ARM in A_M1 B_M2 C_M3; do
    case $ARM in
        A_M1) M=1 ;;
        B_M2) M=2 ;;
        C_M3) M=3 ;;
    esac

    CKPT_DIR=$PROD_DIR/train/arm_$ARM
    KAPPA_LOG=$CKPT_DIR/kappa_history.json
    ARM_LOG=$LOG_DIR/task137_arm_$ARM_${TS}.log

    echo "" | tee -a $LOG_FILE
    echo "===== [Task #137 arm $ARM (M=$M)] start at $(date) =====" | tee -a $LOG_FILE

    CUDA_VISIBLE_DEVICES=1 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
        --M $M \
        --epochs 1000 \
        --batch_size 256 \
        --lr 1e-3 \
        --lr_theta 1e-3 \
        --theta_init 0.01 \
        --kappa_max 0.5 \
        --seed 42 \
        --num_emb_list 64 128 256 \
        --e_dim 32 \
        --layers 512 256 128 \
        --loss_type poincare \
        --beta 1.0 \
        --quant_loss_weight 1.0 \
        --sk_epsilons 0.0 0.0 0.0 \
        --sk_iters 50 \
        --kmeans_init \
        --kmeans_iters 1000 \
        --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
        --device cuda:0 \
        --ckpt_dir $CKPT_DIR \
        --kappa_log_path $KAPPA_LOG \
        --log_interval 10 \
        --save_every 50 \
        2>&1 | tee -a $ARM_LOG

    echo "===== [Task #137 arm $ARM] done at $(date) =====" | tee -a $LOG_FILE
done

echo "" | tee -a $LOG_FILE
echo "===== [Task #137] all 3 arms done at $(date) =====" | tee -a $LOG_FILE

# Clean up
rm -f $PID_FILE
echo "Final ckpts and κ histories in $PROD_DIR/train/" | tee -a $LOG_FILE