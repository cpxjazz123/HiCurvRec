#!/bin/bash
# task4_p0_euclidean_concat_s22.sh — Phase 2 P0.2 Stage 2.2: SID inference (RQ-VAE)
# 注意: P0.2 训练用的是 rqvae_train_flat (有 64-dim encoder bottleneck),
#       所以 inference 必须用 rqvae_inference_flat (匹配训练, codebook [256, 64]),
#       不能用 rkmeans_inference_flat (codebook [256, 928] 不匹配)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
CONCAT_EMB=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt
S21_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task14_p0_euclidean_concat_s21/checkpoints/checkpoint_000_003000.ckpt
OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22

mkdir -p $OUT_DIR

echo "[task58 P0.2 s22] rqvae_inference_flat"
echo "  ckpt: $S21_CKPT"
echo "  embedding: $CONCAT_EMB"
echo "  embedding_dim: 928"
echo "  num_hierarchies: 3"
echo "  out_dir: $OUT_DIR"

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=rqvae_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${CONCAT_EMB} \
    embedding_dim=928 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=${S21_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task58_p0_euclidean_concat_s22 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_euclidean_concat_s22.log

echo "[done] predictions at $OUT_DIR/pickle/"
