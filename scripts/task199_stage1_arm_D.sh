#!/bin/bash
# Task #199 Stage 1 臂 D — c 固定扫描 {1, 10, 30, 100} (4 子臂顺序)
#
# 用户 2026-07-26: 验证"几何不影响性能"正面证据.
# c ∈ {1, 10, 30, 100} 对应 λ ∈ {2.15, 6.7, 30.5, 109}.
# 4 个子臂顺序跑 (同一 GPU, 时间 ×4).
# GPU 3

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task199 $REPO/products/task199

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=3

# 4 子臂
for C_VAL in 1 10 30 100; do
    SAVE_DIR=$REPO/products/task199/stage1_arm_D_c${C_VAL}
    LOG_FILE=$REPO/logs/task199/stage1_arm_D_c${C_VAL}.log
    rm -f $LOG_FILE

    echo "[$(date)] === #199 臂 D: c=${C_VAL}, GPU=$GPU, epochs=1000 ===" | tee -a $LOG_FILE

    CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task199_arm_D_c${C_VAL} \
    python3 -u $REPO/HG-Rec/train_hrqvae.py \
        --data_path $DATA \
        --lr 1e-3 --epochs 1000 --batch_size 256 \
        --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
        --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
        --num_emb_list 64 128 256 --e_dim 32 \
        --beta 0.5 --layers 512 256 128 64 \
        --curvatures ${C_VAL},${C_VAL},${C_VAL} \
        --save_limit 5 \
        --device cuda:0 \
        --ckpt_dir $SAVE_DIR \
        > $LOG_FILE 2>&1
    echo "[$(date)] 子臂 c=${C_VAL} 完成" | tee -a $LOG_FILE
done

echo "[$(date)] === #199 臂 D: 4 子臂全部完成 ===" | tee -a $REPO/logs/task199/stage1_arm_D_done.log