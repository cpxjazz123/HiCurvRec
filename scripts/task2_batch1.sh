#!/bin/bash
# task4_batch1.sh — 启动 task16 批 1（4 个最便宜诊断量）
# CPU only，与 GPU 任务可并行
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID
# -u for unbuffered stdout so log shows progress
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task4_batch1.py