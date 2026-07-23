#!/bin/bash
# task6_q10.sh — 启动 task18 量 10（前缀条件熵）
# CPU only，~10-15 分钟
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID
export OMP_NUM_THREADS=8
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task6_q10.py