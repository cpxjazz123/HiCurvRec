#!/bin/bash
# Task #206i: 欧式 RQ-VAE 单变量 Stage 1 — 只改 --quant_loss_weight=4 (β 保持 baseline=0.5)
#
# 与之前 崩溃版本 的差异:
#   - β=0.5 (paper baseline), 不再用 β=1.0
#   - 其他参数全跟 paper/stage1_baseline_retrain 一样
#   - in-code ×4 (utils.py 已 patch, 也算 baseline)
#
# 唯一变量: --quant_loss_weight = 4 (vs 默认 1.0)
#
# 假设 H: 之前崩 = β=1.0 + in-code ×4 双变量共同问题. 现在单变量应该不崩.
# GPU 0 (空闲)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206i

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
SAVE_DIR=$REPO/products/task206i/stage1_euclidean_qw4_only
LOG_FILE=$REPO/logs/task206i/stage1_single_var_qw4.log
GPU=0

rm -f $LOG_FILE

echo "[$(date)] === Task #206i 单变量 --quant_loss_weight=4 Stage 1 ===" | tee $LOG_FILE
echo "[$(date)]   loss_type=mse, euclidean_qloss=True, β=0.5 (paper), quant_loss_weight=4 (唯一变量)" | tee -a $LOG_FILE
echo "[$(date)]   in-code ×4 in utils.py: (已 patch, 算 baseline)" | tee -a $LOG_FILE
echo "[$(date)]   50 epochs, save_limit=50, eval_step=1, GPU=$GPU" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206i_qw4 \
nohup python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type mse --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 4.0 \
    --euclidean_qloss \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
PID=$!
echo $PID > $REPO/products/task206i/_TRAINING_PID_qw4
echo "[$(date)] PID=$PID, 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE
