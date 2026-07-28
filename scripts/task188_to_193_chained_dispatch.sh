#!/bin/bash
# Chained dispatcher — Task #188 Phase 3 完成时, 自动 fire Phase 4 → β scan → freeze
#
# 检测策略: Task #188 phase 3 (12 stage3_train 进程) 退到 0 → 认为 Phase 3 完成
# 然后串行:
#   1) Phase 4 launcher: scripts/task188_stage4_eval.sh (12 eval + variance matrix)
#   2) β scan 4 臂: scripts/task192_beta_dose_scan.sh
#   3) encoder freeze 1 臂: scripts/task193_encoder_freeze.sh
#
# 用法: nohup ./scripts/task188_to_193_chained_dispatch.sh > logs/chained_dispatch.log 2>&1 &

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_CHAIN=$REPO/logs/chained_dispatch.log
echo "==== Chained dispatcher start at $(date) ====" > $LOG_CHAIN

# 1) Wait for Phase 3 (12 stage3_train 进程) to be 0
echo "[$(date)] Waiting for Task #188 Phase 3 (12 stage3_train processes) to finish..." | tee -a $LOG_CHAIN
while true; do
    PROC_COUNT=$(ps -eo args | grep "task84_hgrec_stage3_train" | grep -v grep | wc -l)
    echo "[$(date)] stage3_train processes alive: $PROC_COUNT" >> $LOG_CHAIN
    if [ "$PROC_COUNT" -le 0 ]; then
        echo "[$(date)] Phase 3 DONE — proceeding to Phase 4" | tee -a $LOG_CHAIN
        break
    fi
    sleep 120  # check every 2 min
done

# 1.5) 确认 12 ckpts 全部存在 (每个 t×s 组合)
TIERS=(t1_8pct t2_10pct t3_12pct t4_13pct)
SEEDS=(42 123 2024)
MISSING=0
for t in "${TIERS[@]}"; do
    for s in "${SEEDS[@]}"; do
        ckpt_dir=$REPO/products/task188/t5small_${t}/seed${s}/Instruments
        best_ckpt=$(ls -t $ckpt_dir/*/HG_Rec_best.pth 2>/dev/null | head -1)
        if [ -z "$best_ckpt" ] || [ ! -f "$best_ckpt" ]; then
            echo "⚠️ Missing ckpt: $t/seed${s}" | tee -a $LOG_CHAIN
            MISSING=$((MISSING+1))
        fi
    done
done
if [ $MISSING -gt 0 ]; then
    echo "❌ $MISSING ckpts missing — aborting Phase 4 dispatch" | tee -a $LOG_CHAIN
    exit 1
fi
echo "✅ All 12 ckpts present — fire Phase 4" | tee -a $LOG_CHAIN

# 2) Fire Phase 4 launcher
bash $REPO/scripts/task188_stage4_eval.sh
EXIT_CODE=$?
echo "[$(date)] Phase 4 exit=$EXIT_CODE" | tee -a $LOG_CHAIN
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Phase 4 failed (exit=$EXIT_CODE). Skip β scan to avoid masking." | tee -a $LOG_CHAIN
    exit 1
fi

# 3) Fire β scan 4 臂 (4 GPU 并发, ~30-45 min)
echo "[$(date)] Fire Task #192 β scan (4 臂)" | tee -a $LOG_CHAIN
bash $REPO/scripts/task192_beta_dose_scan.sh
EXIT_CODE=$?
echo "[$(date)] β scan exit=$EXIT_CODE" | tee -a $LOG_CHAIN
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ β scan failed (exit=$EXIT_CODE). Skip freeze." | tee -a $LOG_CHAIN
    exit 1
fi

# 4) Fire encoder freeze (1 GPU, ~5 min) — 等 β scan 中 1 个先完成以腾出 GPU
echo "[$(date)] Fire Task #193 encoder freeze (1 臂)" | tee -a $LOG_CHAIN
GPU=0  # Use the GPU that became free first
CUDA_VISIBLE_DEVICES=$GPU bash $REPO/scripts/task193_encoder_freeze.sh
EXIT_CODE=$?
echo "[$(date)] freeze exit=$EXIT_CODE" | tee -a $LOG_CHAIN

echo "==== Chained dispatcher complete at $(date) ====" | tee -a $LOG_CHAIN
