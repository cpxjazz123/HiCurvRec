#!/bin/bash
# Task #270 — Stage 1 L0 utilization ≥ 90% curriculum 3 配方验证
# 2026-07-29
#
# 根因:
#   Task #263 直接测 task253 L0=73.44% < 90%, 同机制族 6/6 历史 task 全 < 90%.
#   默认 β=0.5 双曲 commit loss 把码字推到 boundary 让 L0 unique < K=64.
#
# 候选 3 配方 (单一变量):
#   A1: --beta 0.0  (纯欧氏 VQ-VAE, 完全去掉双曲 commit loss)
#   A2: --beta 0.0 + (0 epoch 30 后 kill) --beta 0.5 --init_encoder_from a1
#        (curriculum: 先欧氏散开, 再双曲精修)
#   A3: --beta 0.5 --freeze_encoder_epoch 20
#        (encoder 冻结后只让 codebook+decoder 重分配)
#
# 受控因素:
#   不改 num_emb_list=[64,128,256], e_dim=36, angular_dim=4, radial_dim=32,
#   product_manifold=True, kmeans_init=True, seed=42,
#   Musical_Instruments 5-core (9922 items / 24772 test examples)
#
# 通过条件 (任一配方 PASS):
#   - L0 utilization >= 90% (>=58/64 unique) at epoch 30 以后
#   - L1/L2 utilization >= 80%
#
# R7 GPU: GPU 0 空闲 (Task #269 已 NO-OP 闭合), 4×L40S 全空闲, 不抢
# R12: 训练开始 + 训练结束各 save 一次 (默认行为, hrqvae_trainer.py R12 修复后)
# R8 sub-rule: 不累积 commits

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

# FIX: PYTHONPATH 可能未预设, 用 ${PYTHONPATH:-} 默认空串避免 set -u 触发 unbound variable
export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet
GPU=${GPU:-0}
LOG_DIR=$REPO/logs/task270
mkdir -p $LOG_DIR

# 配方选择: A1 / A2 / A3
RECIPE=${RECIPE:-A1}

case "$RECIPE" in
    A1)
        # 纯欧氏 VQ-VAE: loss_type=mse + β=0.0.
        # 注意: hrqvae.py compute_loss() 只支持 mse/l1/poincare, 没有 "none".
        # train_hrqvae.py argparse 默认 "poincare". 这里用 mse 拿欧氏重构 + β=0 拿无 commit loss.
        SAVE_DIR=$REPO/products/task270/A1_euclidean
        LOSS_TYPE="mse"
        BETA=0.0
        FREEZE_ENC=""
        INIT_FROM=""
        EPOCHS=50
        EVAL_STEP=5
        EXTRA_ARGS=""
        ;;
    A2)
        # Curriculum β: 先 30 epoch loss=none+β=0, 再 warm-start 30 epoch poincare+β=0.5
        echo "[$(date)] A2: 必须先 A1 完成. 检查 $REPO/products/task270/A1_euclidean/"
        SAVE_DIR=$REPO/products/task270/A2_curriculum
        LOSS_TYPE="poincare"
        BETA=0.5
        FREEZE_ENC=""
        # 从 A1 best_collision ckpt 热启动 encoder
        A1_BEST=$(find $REPO/products/task270/A1_euclidean -name "best_collision*.pth" 2>/dev/null | head -1)
        if [ -z "$A1_BEST" ]; then
            echo "[$(date)] ❌ A1 best_collision ckpt 不存在, 请先跑 RECIPE=A1"
            exit 1
        fi
        INIT_FROM="--init_encoder_from $A1_BEST"
        EPOCHS=30
        EVAL_STEP=5
        EXTRA_ARGS=""
        ;;
    A3)
        # poincare loss + β=0.5 + encoder freeze at epoch 20 (Task #193 实现, 不改 src/)
        SAVE_DIR=$REPO/products/task270/A3_freeze_enc
        LOSS_TYPE="poincare"
        BETA=0.5
        FREEZE_ENC="--freeze_encoder_epoch 20"
        INIT_FROM=""
        EPOCHS=50
        EVAL_STEP=5
        EXTRA_ARGS=""
        ;;
    *)
        echo "Usage: RECIPE={A1|A2|A3} $0"
        exit 1
        ;;
esac

mkdir -p $SAVE_DIR
LOG_FILE="$LOG_DIR/stage1_${RECIPE}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "[$(date)] Task #270 L0 utilization curriculum, RECIPE=$RECIPE, beta=$BETA, GPU=$GPU"
echo "[$(date)] 配方: $RECIPE | loss_type=$LOSS_TYPE | beta=$BETA | freeze=$FREEZE_ENC | init_from=$INIT_FROM"
echo "[$(date)] epochs=$EPOCHS | eval_step=$EVAL_STEP"
echo "[$(date)] SAVE_DIR=$SAVE_DIR"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task270_${RECIPE} \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1800 python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs $EPOCHS --batch_size 1024 \
    --loss_type $LOSS_TYPE --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 \
    --e_dim 36 \
    --beta $BETA --layers 512 256 128 36 \
    --product_manifold \
    --angular_dim 4 \
    --radial_dim 32 \
    --assignment_mode_list shared,shared,shared \
    --eval_step $EVAL_STEP \
    --save_limit 1 \
    --device cuda:0 \
    $FREEZE_ENC \
    $INIT_FROM \
    --ckpt_dir $SAVE_DIR \
    > $LOG_FILE 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
echo "---"
echo "[$(date)] step2 monitor lines (L0 utilization):"
grep "step2 monitor ep" $SAVE_DIR/hrqvae.log | tail -10
echo "---"
echo "[$(date)] L0 utilization 提取:"
grep "step2 monitor ep" $SAVE_DIR/hrqvae.log | grep -oE "L0: [^|]*" | tail -10
