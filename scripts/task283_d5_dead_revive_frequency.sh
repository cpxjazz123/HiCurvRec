#!/bin/bash
# Task #283 / D5 — dead_revive frequency 验证 L0 ≥ 90% (Task #282 NO-GO 推论)
# 2026-07-29
#
# 推论 (Task #282): β 不是 L0 ≥ 90% 杠杆 (β=0 → 1.6% mode collapse, β=0.5 → 73.44%).
# L0 ≥ 90% 需要 dead_revive frequency / Sinkhorn curriculum / kmeans 重 init 等等.
# 本任务 (D5) 验证 dead_revive frequency 是否能把 L0 推到 ≥ 90%.
#
# 候选 3 频率 (单一变量):
#   D5-A: --eval_step 5  (默认, task253 baseline)
#   D5-B: --eval_step 1  (高频, 每个 epoch revive)
#   D5-C: --eval_step 10 (低频, 对照)
#
# 受控因素:
#   不改 num_emb_list=[64,128,256], e_dim=36, angular_dim=4, radial_dim=32,
#   product_manifold=True, kmeans_init=True, seed=42,
#   loss_type=poincare, beta=0.5,
#   Musical_Instruments 5-core (9922 items / 24772 test examples)
#
# 通过条件 (任一频率 PASS):
#   - L0 utilization >= 90% (>=58/64 unique) at epoch 30 以后 (pre-revive, 与 task253 口径一致)
#   - L1/L2 utilization >= 80%
#
# R7 GPU: GPU 2 空闲 (Task #279 在 GPU 0/1, 任务 #282 已闭环), 单 GPU × 5-10 min/频率
# R12: 每 N epoch (eval_step) 自动 save (默认行为)
# R8 sub-rule: 不累积 commits — 本 tick 单 launcher, 单 commit

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

# FIX: PYTHONPATH 可能未预设, 用 ${PYTHONPATH:-} 默认空串避免 set -u 触发 unbound variable
export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047_ar57_scratch/wenyu/hf_models}

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=${GPU:-2}
LOG_DIR=$REPO/logs/task283
mkdir -p $LOG_DIR

RECIPE=${RECIPE:-D5A}
EVAL_STEP=${EVAL_STEP:-5}

case "$RECIPE" in
    D5A)
        SAVE_DIR=$REPO/products/task283/A_eval5
        EVAL_STEP=5
        ;;
    D5B)
        SAVE_DIR=$REPO/products/task283/B_eval1
        EVAL_STEP=1
        ;;
    D5C)
        SAVE_DIR=$REPO/products/task283/C_eval10
        EVAL_STEP=10
        ;;
    *)
        echo "Usage: RECIPE={D5A|D5B|D5C} $0"; exit 1 ;;
esac

mkdir -p $SAVE_DIR
LOG_FILE="$LOG_DIR/stage1_${RECIPE}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "[$(date)] Task #283 D5 dead_revive frequency, RECIPE=$RECIPE, eval_step=$EVAL_STEP, GPU=$GPU"
echo "[$(date)] SAVE_DIR=$SAVE_DIR"
echo "[$(date)] LOSS=poincare BETA=0.5 anti_collapse=dead_revive eval_step=$EVAL_STEP"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task283_${RECIPE} \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1800 python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 50 --batch_size 1024 \
    --loss_type poincare --beta 0.5 \
    --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode_list shared,shared,shared \
    --anti_collapse dead_revive \
    --eval_step $EVAL_STEP \
    --save_limit 1 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
echo "---"
echo "[$(date)] step2 monitor lines (L0 utilization pre-revive):"
find $SAVE_DIR -name "hrqvae.log" | head -1 | xargs -I {} grep "step2 monitor ep" {} | tail -10
echo "---"
echo "[$(date)] L0 utilization pre-revive 提取:"
find $SAVE_DIR -name "hrqvae.log" | head -1 | xargs -I {} grep "step2 monitor ep" {} | grep -oE "L0: [^|]*" | tail -10
echo "---"
echo "[$(date)] dead_revive post-revive 提取:"
find $SAVE_DIR -name "hrqvae.log" | head -1 | xargs -I {} grep "dead_revive L" {} | tail -10
