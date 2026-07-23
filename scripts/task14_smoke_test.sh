#!/bin/bash
# task4_smoke_test.sh — Task #437 Step 1: 验证 h_offset 改造 + 5 个 variant smoke test
# 200 step each, ~3 min per variant, total ~15 min GPU
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
mkdir -p "$LOG_DIR"

# 5 个 variant smoke test
# A: h_dim=0 取消 H (应 ~ P0.2)
# P0.3: h_offset=0, h_dim=32 (baseline)
# B: h_offset=768, h_dim=32 (H 在 brand)
# C: h_offset=800, h_dim=96 (H 在 taxonomy)
# D: h_offset=896, h_dim=32 (H 在 behavior)
for variant in "0:0" "0:32" "768:32" "800:96" "896:32"; do
    IFS=':' read -r offset dim <<< "$variant"
    name="smoke_hoff${offset}_hdim${dim}"
    log="$LOG_DIR/task56_${name}.log"
    echo "[task56 smoke] starting variant h_offset=$offset h_dim=$dim"
    CUDA_VISIBLE_DEVICES=0 python3 task_artifacts/scripts/mixed_curvature/task4_h_offset_train.py \
        --h_offset=$offset \
        --h_dim=$dim \
        --max-steps=200 \
        --log-every=50 \
        --task-tag="task56_smoke" \
        2>&1 | tee "$log"
    echo "[task56 smoke] done variant $name, ckpt at logs/train/runs/task56_smoke_${name}/"
done

echo "[task56 smoke] all 5 variants done. 检查 logs/train/runs/ 下 ckpt 是否齐全"