#!/bin/bash
# Task #206j 诊断: 两臂各跑 5 epochs, 对比 recon_loss vs rq_loss 比值
# 算出欧式需要的 quant_loss_weight
#
# 配置: 两臂都用 qw=1.0 (paper baseline), 双曲 β=1.0 (Task #84), 欧式 β=0.5 (paper)
#
# 输出目标: log 里找 "DIAG[ep0 batch0]" 行, 拿 recon / rq_loss / 比值
#
# GPU 0: 双曲 baseline, GPU 2: 欧式 baseline

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206j $REPO/products/task206j

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

# === 臂 A: 双曲 baseline (β=1.0) ===
echo "[$(date)] === 臂 A 双曲 baseline (β=1.0) — GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206j_hyp \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 5 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 1.0 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --save_limit 1 \
    --device cuda:0 \
    --ckpt_dir $REPO/products/task206j/hyp_diag \
    > $REPO/logs/task206j/hyp_diag.log 2>&1 &
HYP_PID=$!
echo "[$(date)] 双曲 PID=$HYP_PID"

# === 臂 B: 欧式 baseline (β=0.5, in-code ×4 已有, euclidean_qloss, qw=1.0) ===
sleep 3
echo "[$(date)] === 臂 B 欧式 baseline (β=0.5, qw=1.0) — GPU 2 ==="
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206j_euc \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 5 --batch_size 1024 \
    --loss_type mse --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --euclidean_qloss \
    --save_limit 1 \
    --device cuda:0 \
    --ckpt_dir $REPO/products/task206j/euc_diag \
    > $REPO/logs/task206j/euc_diag.log 2>&1 &
EUC_PID=$!
echo "[$(date)] 欧式 PID=$EUC_PID"

echo "$HYP_PID" > $REPO/products/task206j/_PID_hyp
echo "$EUC_PID" > $REPO/products/task206j/_PID_euc

# 等两个都跑完
wait $HYP_PID 2>/dev/null || true
wait $EUC_PID 2>/dev/null || true

echo "[$(date)] === 两臂 5 epoch 完成 ==="

echo ""
echo "=== 臂 A 双曲 epoch 0 diagnostic 段 ==="
grep "DIAG\[ep0" $REPO/logs/task206j/hyp_diag.log | head -12

echo ""
echo "=== 臂 B 欧式 epoch 0 diagnostic 段 ==="
grep "DIAG\[ep0" $REPO/logs/task206j/euc_diag.log | head -12
