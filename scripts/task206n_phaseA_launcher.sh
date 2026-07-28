#!/bin/bash
# Task #206n — Phase A: usage-target_r 半径语义项扫描 (3 臂 × 500 epoch)
#
# 变量: w_rad ∈ {0, 0.1, 1.0} (对照组 / 轻 / 重)
# 不变: β=0.5, num_emb_list=[64,128,256], e_dim=32, loss_type=poincare
#       sk_epsilons=[0,0,0] (paper 默认, 关闭 Sinkhorn)
#       per-layer c: 93 (L0), 604 (L1), 702 (L2) — 目标强度 0.85 (用户 2026-07-26)
#
# Patch 位置:
#   1. HG-Rec/model/utils.py: HVectorQuantization.__init__ + forward 加 L_rad
#   2. HG-Rec/model/utils.py: HResidualVectorQuantization.__init__ 加 w_rad_list
#   3. HG-Rec/model/hrqvae.py: HRQVAE.__init__ 透传 w_rad_list
#   4. HG-Rec/train_hrqvae.py: 加 --w_rad CLI + pass-through
#
# 预算: ~6-8 h (500 epoch × 3 GPU 并发)
#
# SBATCH 参数 (3 个独立任务, 各占 1 GPU)
#SBATCH --job-name=task206n_phaseA
#SBATCH --output=logs/task206n/slurm_%A_%a.log
#SBATCH --error=logs/task206n/slurm_%A_%a.log
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206n $REPO/products/task206n

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

# per-layer c: L0=93, L1=604, L2=702 (目标强度 0.85)
CURVATURES="93,604,702"

# 3 个 w_rad 臂 (分配到 SLURM_ARRAY_TASK_ID 0/1/2)
W_RADS=(0.0 0.1 1.0)
W_RAD=${W_RADS[$SLURM_ARRAY_TASK_ID]}

# 如果不用 SBATCH, 用 GPU ID 做臂分配
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # 非 SLURM 环境: 用 GPU 编号区分
    GPU_ID=${1:-0}
    case $GPU_ID in
        0) W_RAD=0.0 ;;
        1) W_RAD=0.1 ;;
        2) W_RAD=1.0 ;;
        *) echo "Usage: $0 <gpu_id> (0|1|2)"; exit 1 ;;
    esac
fi

LOG=$REPO/logs/task206n/arm_wrad${W_RAD}_stage1.log
SAVE_DIR=$REPO/products/task206n/wrad${W_RAD}

echo "[$(date)] === Task #206n Phase A: w_rad=${W_RAD}, c=${CURVATURES} ==="
echo "[$(date)] beta=0.5, epochs=500, batch_size=1024"
echo "[$(date)] log=$LOG, save=$SAVE_DIR"

# R12: 训练前确保目录存在 + 写 PID
mkdir -p $SAVE_DIR
echo $$ > $SAVE_DIR/_TRAINING_PID

# 如果是 SLURM, 环境已设 CUDA_VISIBLE_DEVICES; 否则手动设
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${1:-0}
fi

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206n_wrad${W_RAD} \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 500 --batch_size 1024 \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures $CURVATURES \
    --quant_loss_weight 1.0 \
    --w_rad $W_RAD \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG 2>&1

# 训练结束后清理 PID
rm -f $SAVE_DIR/_TRAINING_PID

echo "[$(date)] === Task #206n Phase A arm w_rad=${W_RAD} 完成 ==="
