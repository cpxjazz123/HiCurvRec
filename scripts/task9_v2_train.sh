#!/bin/bash
# task1_v2_train.sh — Task 15 v2 对照实验：训练时密集保存 ckpt（修复 save_top_k）
#
# 关键改动（v2 修复版）：
#   - callbacks.model_checkpoint.every_n_train_steps=100（每 100 步存一个 ckpt）
#   - callbacks.model_checkpoint.save_top_k=-1（保留所有 ckpt，不只 best）
#   - callbacks.model_checkpoint.save_last=true（同时存 last.ckpt）
#   - 注：monitor/mode **不能**显式设为 null（Lightning 拒绝 mode=None 会抛 MisconfigurationException）
#     ⇒ 改用 yaml 默认 monitor=train/loss, mode=min（save_top_k=-1 时 monitor 实际不生效但必须合法）
#   - num_hierarchies=4
#   - max_steps=1600（4×400 步 layer-wise；每层 400 步 → 每层 4 个 ckpt）
#
# 数据：复用 task.md Stage 1 embedding
# 输出：16 个 ckpt 在 logs/train/runs/task11_v2_dense_ckpts_v2/checkpoints/
#
# 启动命令：
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task1_v2_train.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=2 python -m src.train \
    experiment=rkmeans_train_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=4 \
    codebook_width=256 \
    trainer.max_steps=1600 \
    callbacks.model_checkpoint.every_n_train_steps=100 \
    callbacks.model_checkpoint.save_top_k=-1 \
    callbacks.model_checkpoint.save_last=true \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task11_v2_dense_ckpts_v3 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task1_v2_train_v2.log