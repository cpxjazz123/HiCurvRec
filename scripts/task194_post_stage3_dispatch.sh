#!/bin/bash
# Task #194 — Post-Stage 3 dispatcher (Stage 4 + K0 diagnosis + verdict)
# 等 4 臂 T5-mini Stage 3 完成 → fire K0 L0_err diagnosis + verdict aggregate
# Stage 4 (test R@10 eval) 等 Stage 3 best_ckpt 落盘后手动 fire 或合并到 verdict aggregate

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG=$REPO/logs/task194/post_dispatch.log
echo "==== Task #194 post-Stage 3 dispatch start at $(date) ====" > $LOG

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# 1) Wait for 4 Stage 3 main processes to finish
echo "[$(date)] Waiting for 4 Stage 3 task84_hgrec_stage3_train main processes to finish..." | tee -a $LOG
while true; do
    MAIN_COUNT=$(ps -eo args | grep "task84_hgrec_stage3_train" | grep "task194" | grep -v grep | wc -l)
    echo "[$(date)] stage3(task194) main alive: $MAIN_COUNT" >> $LOG
    if [ "$MAIN_COUNT" -le 0 ]; then
        echo "[$(date)] Stage 3 DONE — proceeding to K0 L0_err diagnosis" | tee -a $LOG
        break
    fi
    sleep 120  # check every 2 min
done

# 2) K0 L0_err diagnosis (单 GPU, 复用 task194_k0_l0_err_diagnose.py)
echo "[$(date)] Fire K0 L0_err diagnosis" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 python3 -u $REPO/scripts/task194_k0_l0_err_diagnose.py > $REPO/logs/task194/k0_l0_err.log 2>&1 || echo "⚠️ K0 L0_err diagnose failed" | tee -a $LOG

# 2.5) Task #188 后续 — 12 ckpt val set 评估 (用户 2026-07-25 23:04 要求报 val 指标, val metrics 在训练中丢失)
echo "[$(date)] Fire Task #188 val metrics 补救 (12 ckpt × val set)" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 python3 -u $REPO/scripts/task188_val_eval.py > $REPO/logs/task188/val_eval.log 2>&1 || echo "⚠️ task188 val eval failed" | tee -a $LOG

# 3) Stage 4 + verdict aggregate (从 Stage 3 val log 抽 R@10)
echo "[$(date)] Aggregate verdict → verdicts/task194_k0_capacity_result.md" | tee -a $LOG
python3 -u $REPO/scripts/task194_aggregate_verdict.py > $REPO/logs/task194/verdict.log 2>&1 || echo "⚠️ verdict aggregate failed" | tee -a $LOG

echo "==== Task #194 post-Stage 3 dispatch complete at $(date) ====" | tee -a $LOG