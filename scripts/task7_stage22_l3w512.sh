#!/bin/bash
# task9_stage22_l3w512.sh — task11 L=3, W=512 Stage 2.2 RKMeans inference
#
# GPU 分配: GPU 1 (cuda:1) — task11 l3w512 Stage 2.1 完成后腾出
# 输入: Stage 2.1 checkpoint
# 输出: <run_ts>/pickle/merged_predictions_tensor.pt  (shape: (4, 11924))
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_stage22_l3w512.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
CKPT_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-10/12-49-50/checkpoints/checkpoint_000_003000.ckpt

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=1 python -m src.inference \
    experiment=rkmeans_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=512 \
    ckpt_path=${CKPT_PATH} \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task9_s22_l3w512.log