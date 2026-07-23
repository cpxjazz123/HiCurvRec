#!/bin/bash
# task4_p0_euclidean_concat_s21.sh — Phase 2 P0.2 Stage 2.1: Euclidean-Concat RQ-VAE
# 输入: 928-dim multi-factor concat embedding (text+brand+taxonomy+behavior)
# 与 task13_group_a_s21 配对: rqvae_train_flat + 3000 step + codebook 256 + L=3
# 输出: ckpt → task4_p0_euclidean_concat_s22.sh 推断
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

CONCAT_EMB=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt
DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

echo "[task58 P0.2] Euclidean-Concat 928-dim RQ-VAE"
echo "  embedding: ${CONCAT_EMB}"
echo "  embedding_dim: 928"
echo "  num_hierarchies: 3"
echo "  codebook_width: 256"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=rqvae_train_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${CONCAT_EMB} \
    embedding_dim=928 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task58_p0_euclidean_concat_s21 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_euclidean_concat_s21.log

echo "[done] ckpt at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task58_p0_euclidean_concat_s21/checkpoints/"