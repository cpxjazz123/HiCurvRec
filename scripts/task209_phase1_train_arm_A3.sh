#!/bin/bash
# Task #209 Phase 1 臂 A3 — 完整方法 (--radii + --scale_norm + --w_path 1.0 hyp + rho_targets_path)
#
# 核心创新: 路径正则化 loss += w_path · path_cost([c0, c1, c2], hyperbolic=True).
# 用 Poincaré 距离算测地线偏移, 鼓励 per-layer 码字沿测地线排列.
# 这是 Phase 1 主推臂, 必须与 A4 对照才能证明几何作用.
#
# 配置: A2 + --w_path 1.0 --path_geometry hyp, GPU 2
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task209 $REPO/products/task209

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=2
SAVE_DIR=$REPO/products/task209/phase1_arm_A3
LOG_FILE=$REPO/logs/task209/phase1_arm_A3.log
rm -f $LOG_FILE

echo "[$(date)] === Task #209 Phase 1 臂 A3: 完整方法 (radii+scale_norm+w_path=1.0 hyp), GPU=$GPU ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task209_arm_A3 \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 1000 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --norm_target 1.0 1.35 1.70 \
    --gamma_norm 0.1 \
    --scale_norm poincare \
    --w_path 1.0 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 \
    --save_limit 5 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task209/_TRAINING_PID_arm_A3
echo "[$(date)] Arm A3 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE
