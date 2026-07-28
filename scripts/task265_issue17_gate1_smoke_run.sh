#!/bin/bash
# Task #265 / Issue #17 Gate 1 (a) — verify utilization print fix
# ≤3 epoch smoke run, eval_step=1, no init_encoder_from
# 必须真打印 "[step2 monitor ep1] collision_rate=... | L0: r_std=... usage=... (K/M) | ..." 这种三行

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/products/task265/hrqvae_smoke_test $REPO/logs/task265

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=0
SAVE_DIR=$REPO/products/task265/hrqvae_smoke_test
LOG_FILE=$REPO/logs/task265/smoke_stage1_train.out
rm -f $LOG_FILE

echo "[$(date)] === Task #265 (Issue #17 Gate 1 a): 3 epoch smoke run, GPU=$GPU ===" | tee $LOG_FILE
echo "[$(date)] expected: hrqvae.log 里有 step2 monitor ep1/ep2/ep3 三行 utilization" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task265_smoke \
timeout 600 python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 3 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta 0.5 --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode_list shared,shared,shared \
    --eval_step 1 \
    --save_limit 1 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1
EXIT=$?
echo "[$(date)] exit code=$EXIT" | tee -a $LOG_FILE
echo "---"
echo "[$(date)] check step2 monitor output:"
grep "step2 monitor ep" $SAVE_DIR/hrqvae.log | tail -10
echo "---"
echo "[$(date)] check fail logs (before fix we'd see UnboundLocalError):"
grep "UnboundLocalError\|step2 monitor.*logging failed" $SAVE_DIR/hrqvae.log | tail -5 || echo "(none → fix PASS)"