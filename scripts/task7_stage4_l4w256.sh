#!/bin/bash
# task9_stage4_l4w256.sh — task11 L=4, W=256 Stage 4 inference (Encoder-Decoder)
#
# GPU 分配: GPU 1 (cuda:1)
# 输入: best ckpt (step 3375, val/recall@10=0.07676) + SID dedup tensor (shape (5, 11924))
# 输出: <run_ts>/pickle/merged_predictions_tensor.pt  (semantic_ids predictions)
#       然后用 task6_constrained_eval.py 评估 Recall@5/10, NDCG@5/10
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_stage4_l4w256.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-10/13-28-19/pickle/merged_predictions_tensor_dedup.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task9_l4w256.ckpt

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=1 python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=5 \
    sequence_length=120 \
    seed=42 \
    ckpt_path=${CKPT_PATH} \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task9_l4w256_s4 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task9_s4_l4w256.log