#!/bin/bash
# task7_v3_stage4_inference.sh — task9 v3 Decoder-Only Stage 4 inference
#
# GPU 分配: GPU 0 (cuda:0) — task9 v3 训练完成后腾出
# 输入: task9 v3 训练 checkpoint + SID dedup tensor
# 输出: <run_ts>/pickle/merged_predictions_tensor.pt  (semantic_ids predictions)
#       然后用 task7_dec_only_eval.py 评估 Recall@5/10, NDCG@5/10
#
# 启动命令:
#   cd /fs04/ar57/wenyu/GeneRec/GRID
#   bash /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task7_v3_stage4_inference.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-09/09-28-56/pickle/merged_predictions_tensor_dedup.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
CKPT_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-10/12-49-47/checkpoints/checkpoint_epoch=000_step=000250.ckpt

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=tiger_decoder_only_inference_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    seed=42 \
    ckpt_path=${CKPT_PATH} \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task7_v3_s4_inference.log