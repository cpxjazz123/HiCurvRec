#!/bin/bash
# task6_diag_a_d3.sh — Task #67 深度诊断 A+D3 启动脚本
#
# Exp A: 未归一化原生空间下三距离最近邻一致性测试
# Exp D3: 各层残差信息通过率分析
#
# 启动命令:
#   cd /home/wlia0047/ar57/wenyu/GeneRec
#   bash scripts/task6_diag_a_d3.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task16/from_task67
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

# ====== Exp A ======
echo "=========================================="
echo "[Task #67] Exp A: native space equivalence"
echo "=========================================="
python scripts/task6_exp_a_native_space.py \
    --stage1_pt /fs04/ar57/wenyu/GeneRec/logs/inference/runs/2026-07-16/21-56-20/pickle/merged_predictions_tensor.pt \
    --rqvae_ckpt /fs04/ar57/wenyu/GeneRec/logs/train/runs/2026-07-16/21-59-42/checkpoints/checkpoint_000_015000.ckpt \
    --out_json products/task16/from_task67/exp_a_native_space.json 2>&1 \
    | tee GRID/task_artifacts/scripts/logs/task6_exp_a.log

# ====== Exp D3 ======
echo "=========================================="
echo "[Task #67] Exp D3: info throughput"
echo "=========================================="
python scripts/task6_exp_d3_info_throughput.py \
    --stage1_pt /fs04/ar57/wenyu/GeneRec/logs/inference/runs/2026-07-16/21-56-20/pickle/merged_predictions_tensor.pt \
    --rqvae_ckpt /fs04/ar57/wenyu/GeneRec/logs/train/runs/2026-07-16/21-59-42/checkpoints/checkpoint_000_015000.ckpt \
    --out_json products/task16/from_task67/exp_d3_info_throughput.json 2>&1 \
    | tee GRID/task_artifacts/scripts/logs/task6_exp_d3.log

echo "[Task #67] Both experiments finished. See:"
echo "  - products/task16/from_task67/exp_a_native_space.json"
echo "  - products/task16/from_task67/exp_d3_info_throughput.json"