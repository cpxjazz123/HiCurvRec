#!/bin/bash
# idea1_WF_s21.sh — Idea 1 现象 2: 按 WF 极端分配 [256, 64, 16] 重训 RQ
# 与 AQ baseline (task13_group_a_s21) 唯一差异: codebook_width per-layer
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=rkmeans_train_wf \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.normalize_residuals=false \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=idea1_WF_s21_K256_64_16 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/idea1_WF_s21.log
