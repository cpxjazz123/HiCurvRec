#!/bin/bash
# task3_group_c_s21.sh — Group C (GSRQ gain-shape fusion) Stage 2.1 RKMeans 训练
#
# 关键参数:
#   - experiment=rkmeans_train_gsrq (新 yaml, inherit rkmeans_train_flat + 启用 GSRQ)
#   - trainer.max_steps=3000 trainer.val_check_interval 保持 yaml 默认值
#   - 只用 1 个 GPU (dev 1, 与 Group A Stage 3 在 dev 0 并行)
#
# 输出: logs/train/runs/${id}/checkpoints/checkpoint_000_003000.ckpt

set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

CUDA_VISIBLE_DEVICES=1 python -m src.train \
    experiment=rkmeans_train_gsrq \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.max_steps=3000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task13_group_c_s21 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_c_s21.log
