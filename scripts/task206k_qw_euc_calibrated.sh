#!/bin/bash
# Task #206k: 欧式 RQ-VAE 用 ratio-derived quant_loss_weight 验证
#
# qw_euc = ratio_hyp / ratio_euc = 5.86e-04 (从 task206j 实测 recon/quant 比值反推)
#
# 仅变量: --quant_loss_weight 1.0 → 5.86e-04
# 其它全不动: β=0.5, in-code ×4 (utils.py), euclidean_qloss, num_emb_list=[64,128,256]

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206k $REPO/products/task206k
DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

LOG=$REPO/logs/task206k/stage1_qw_euc_calibrated.log
GPU=0

echo "[$(date)] === Task #206k 欧式 qw=5.86e-04 (ratio-derived) ==="
echo "[$(date)] β=0.5, in-code ×4 保留, euclidean_qloss, qw = 5.86e-04"
echo "[$(date)] 期望: 不再 mode collapse, 跟双曲曲线比较"

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206k \
nohup python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type mse --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 0.000586 \
    --euclidean_qloss \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $REPO/products/task206k/euc_qw_calibrated \
    > $LOG 2>&1 &
PID=$!
echo $PID > $REPO/products/task206k/_TRAINING_PID
echo "[$(date)] PID=$PID, GPU=$GPU, log=$LOG"
