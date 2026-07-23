#!/bin/bash
# task6_q11.sh — 启动 task18 量 11（跨层码本失配 Δ_l）
# CPU only，复用 task16 cache
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID
export OMP_NUM_THREADS=8
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task6_q11.py