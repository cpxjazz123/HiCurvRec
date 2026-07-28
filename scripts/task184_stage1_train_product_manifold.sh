#!/bin/bash
# Task #184 Stage 1 — Phase 1a product manifold RQ-VAE (球面 x 双曲解耦码字)
# 修复 Task #183 v2 暴露的角向坍缩问题. 
# angular_dim=16 (球面) + radial_dim=16 (双曲径向), alpha=1.0, beta_radial=1.0
# GPU 0

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047_ar57_scratch/wenyu/hf_models 2>/dev/null || export HF_HOME=/home/wlia0047/.cache/huggingface

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task184
mkdir -p /fs04/ar57/wenyu/GeneRec/logs/task184

echo "===== [Task #184 Stage 1] Product manifold RQ-VAE at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --radii 1.0 1.345 1.69 \
    --rho_reg_weight 0.1 \
    --product_manifold \
    --angular_dim 16 \
    --radial_dim 16 \
    --alpha 1.0 \
    --beta_radial 1.0 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task184/hrqvae_product_manifold \
    --device cuda:0 \
    2>&1 | tee /fs04/ar57/wenyu/GeneRec/logs/task184/stage1_product_manifold.out

echo "===== [Task #184 Stage 1] completed at $(date) ====="
