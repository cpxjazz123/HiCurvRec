#!/usr/bin/env bash
# Task #324 / Issue #39 — Stage 4 召回改造 5-arm 串行 launch
# 5 Arms: A_hnsw / B_ivf_pq / C_rerank / D_beam100 / D_beam200 / E_control
# ckpt = task243 Stage 3 ckpt + Issue #30 SID endpoint
# R7: 等 task320 释放 GPU, 串行跑 (避免 5 个同时启动抢卡)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

mkdir -p logs/task324

CKPT=$REPO/products/task243/t5mini_epoch200/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth

# 5-arm serial launch on GPU 1 (assuming task320 GPU 0/2/3 done first, GPU 1 free)
for arm in E_control A_hnsw B_ivf_pq C_rerank D_beam100 D_beam200; do
    LOG=logs/task324/stage4_${arm}.log
    echo "=== task324 ${arm} ===" | tee "$LOG"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/task324_issue39_stage4_5arm_eval.py \
        --arm "$arm" --ckpt_path "$CKPT" --gpu 0 \
        2>&1 | tee -a "$LOG"
done
echo "=== all 6 evals done ==="
