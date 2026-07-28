#!/bin/bash
# Task #189 diagnose all baseline ckpts
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_fix_v2/Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000"

# (1) 最低 collision (epoch 59, 9.05%)
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task189_codebook_geometry_diagnose.py \
    --ckpt_path "$CKPT_DIR/epoch_59_collision_0.0905_model.pth" \
    --label "Task181_LOWEST_epoch59_coll9.05pct" \
    --device cuda:0 \
    > /home/wlia0047/ar57/wenyu/GeneRec/logs/task189/01_lowest.out 2>&1

# (2) 最终 collision (epoch 999, 12.39%)
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task189_codebook_geometry_diagnose.py \
    --ckpt_path "$CKPT_DIR/epoch_999_collision_0.1239_model.pth" \
    --label "Task181_FINAL_epoch999_coll12.39pct" \
    --device cuda:0 \
    > /home/wlia0047/ar57/wenyu/GeneRec/logs/task189/02_final.out 2>&1

echo "DONE — Task #189 full diagnosis"
