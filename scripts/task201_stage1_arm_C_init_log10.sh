#!/bin/bash
# Task #201 Stage 1 臂 C — exp_per_layer with θ_init=log(10) (用户拍板选项 A)
#
# 跟 #199 臂 C 唯一区别: theta_init=2.3026 (c_init=10.0) 而非 #199 默认 0.0 (c_init=1.0).
# 验证用户预测: 三层 θ_init=log(10) 后, 每层 θ_ℓ 学起来都 > log(10) (c_ℓ > 10) 而非 #199 学到 0.72/0.93/0.97 (向下).
#
# 配置 (跟 #199 C / #181 baseline 对齐):
#   epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
#   lr=1e-3, kmeans_iters=1000, loss_type=poincare
# 变量 (vs #199 C): theta_init=log(10)=2.3026
# GPU 2

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task201 $REPO/products/task201

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task201/stage1_arm_C_exp_per_layer_init_log10
LOG_FILE=$REPO/logs/task201/stage1_arm_C.log
rm -f $LOG_FILE

THETA_INIT=$(python3 -c "import math; print(math.log(10))")
echo "[$(date)] === #201 臂 C: exp_per_layer, θ_init=log(10)=$THETA_INIT (c_init=10.0), GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task201_arm_C \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 256 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --kappa_mode exp_per_layer --r_target_list 2.0,2.7,3.4 \
    --theta_init $THETA_INIT \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task201/_TRAINING_PID_arm_C
echo "[$(date)] Arm C launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE