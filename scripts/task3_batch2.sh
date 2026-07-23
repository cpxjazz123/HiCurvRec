#!/bin/bash
# task5_batch2.sh — 启动 task17 批 2（3 个跨层诊断量）
# CPU only，与 GPU 任务可并行
# 重用 task16 _rqidx.pt cache（避免重做 forward）
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID
# 控制线程避免冲突
export OMP_NUM_THREADS=8
python3 -u /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/task5_batch2.py