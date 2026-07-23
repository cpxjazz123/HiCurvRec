#!/bin/bash
# idea1_WF_s22.sh — Idea 1 WF Stage 2.2 inference (用 WF ckpt 推 SID)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
CKPT=/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/idea1_WF_s21_K256_64_16/checkpoints/checkpoint_000_003000.ckpt

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=rkmeans_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=${CKPT} \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=idea1_WF_s22_K256_64_16 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/idea1_WF_s22.log
