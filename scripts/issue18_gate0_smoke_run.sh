#!/bin/bash
# Issue #18 Gate 0 (b) — ≤3 epoch smoke run 验证 step2 monitor 真打印
# 2026-07-29
#
# 通过条件 (Issue #18 Gate 0 (b)):
#   - hrqvae.log 出现 [step2 monitor ep1]
#   - 三层 usage 数字 (L0/L1/L2) 齐全
#   - UnboundLocalError 计数 = 0
#
# 设计: 3 epoch euclidean loss (避开 Sinkhorn 噪声, 只验证 trainer 行 495 step2 monitor 真打印).
# GPU 2 空闲 (GPU 0/1 被 Task #279 占). 估约 50s 跑完.

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
LOG=$REPO/logs/issue18_gate0_smoke.log
OUT_DIR=$REPO/products/issue18_gate0_smoke
mkdir -p $REPO/logs $OUT_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH

TRITON_CACHE=/home/wlia0047/.triton/cache_issue18_smoke
mkdir -p $TRITON_CACHE

echo "[$(date)] === Issue #18 Gate 0 (b) smoke run ===" | tee $LOG
echo "    epochs=3 loss_type=poincare (3 epoch 估约 50s 验证 trainer 行 495 step2 monitor)" | tee -a $LOG
echo "    OUT_DIR=$OUT_DIR GPU 2 (空闲)" | tee -a $LOG

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=$TRITON_CACHE DISABLE_USAGE_KILL=1 \
    python3 -u train_hrqvae.py \
        --epochs 3 \
        --warmup_epochs 1 \
        --batch_size 256 \
        --num_workers 2 \
        --eval_step 1 \
        --learner AdamW \
        --lr_scheduler_type linear \
        --data_path ./dataset/Instruments/item_emb.parquet \
        --weight_decay 0.0 \
        --dropout_prob 0.0 \
        --loss_type poincare \
        --kmeans_init True \
        --kmeans_iters 50 \
        --sk_epsilons 0.0 0.0 0.0 \
        --sk_iters 30 \
        --num_emb_list 64 128 256 \
        --e_dim 32 \
        --quant_loss_weight 1.0 \
        --beta 0.5 \
        --layers 512 256 128 64 \
        --save_limit 50 \
        --device cuda:0 \
        --ckpt_dir $OUT_DIR \
        >> $LOG 2>&1
echo "[$(date)] === Smoke run DONE ===" | tee -a $LOG

# ===== Gate 0 (b) 校验 =====
HRQVAE_LOG=$OUT_DIR/hrqvae.log
echo "----- Gate 0 (b) Verification -----" | tee -a $LOG

if [ ! -f "$HRQVAE_LOG" ]; then
    echo "❌ FAIL: hrqvae.log not found at $HRQVAE_LOG" | tee -a $LOG
    exit 1
fi

# 1. step2 monitor 真打
STEP2_COUNT=$(grep -c "\[step2 monitor ep" "$HRQVAE_LOG" 2>/dev/null || echo 0)
echo "Step2 monitor line count: $STEP2_COUNT (期望 ≥ 1)" | tee -a $LOG
[ "$STEP2_COUNT" -ge 1 ] || { echo "❌ FAIL: step2 monitor 未打印"; exit 1; }

# 2. 三层 usage 数字齐全 (任何一行 step2 monitor)
USAGE_LINES=$(grep "usage=" "$HRQVAE_LOG" | head -1)
echo "First usage line: $USAGE_LINES" | tee -a $LOG
# 检查 L0 L1 L2 (用冒号或 L? 模式)
L0=$(grep "usage=" "$HRQVAE_LOG" | head -1 | grep -oE "L0:[^ ]*" || echo "MISSING")
L1=$(grep "usage=" "$HRQVAE_LOG" | head -1 | grep -oE "L1:[^ ]*" || echo "MISSING")
L2=$(grep "usage=" "$HRQVAE_LOG" | head -1 | grep -oE "L2:[^ ]*" || echo "MISSING")
echo "  $L0 | $L1 | $L2" | tee -a $LOG
{ [ "$L0" != "MISSING" ] && [ "$L1" != "MISSING" ] && [ "$L2" != "MISSING" ]; } || { echo "❌ FAIL: 三层 usage 缺失"; exit 1; }

# 3. UnboundLocalError 计数
UNBOUND_COUNT=$(grep -c "UnboundLocalError\|referenced before assignment" "$HRQVAE_LOG" 2>/dev/null || echo 0)
echo "UnboundLocalError count: $UNBOUND_COUNT (期望 = 0)" | tee -a $LOG
[ "$UNBOUND_COUNT" -eq 0 ] || { echo "❌ FAIL: trainer 仍有 UnboundLocalError"; exit 1; }

echo "✅ Issue #18 Gate 0 (b) PASS: step2 monitor 真打印, 三层 usage 齐全, UnboundLocalError=0" | tee -a $LOG
