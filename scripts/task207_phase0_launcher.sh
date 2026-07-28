#!/bin/bash
# Task #207 — Phase 0: 欧式 vs 双曲 collision 对齐 Stage 1 (2 臂 × 500 epoch 并发)
#
# 臂 H: 双曲 (HG-Rec baseline) — loss_type=poincare, euclidean_qloss=False
# 臂 E: 欧式 — loss_type=mse, euclidean_qloss=True
#
# SBATCH 参数 (array=0-1 → 臂 H + 臂 E, 各占 1 GPU L40S)
#SBATCH --job-name=task207_euc_vs_hyp
#SBATCH --output=logs/task207/slurm_%A_%a.log
#SBATCH --error=logs/task207/slurm_%A_%a.log
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task207 $REPO/products/task207

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

# 2 臂定义
NAMES=(hyp euc)
LOSS_TYPES=(poincare mse)
EUCLIDEAN_QLOSS=(False True)

SLOT=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$SLOT]}
LOSS_TYPE=${LOSS_TYPES[$SLOT]}
EUC_Q=${EUCLIDEAN_QLOSS[$SLOT]}

LOG=$REPO/logs/task207/arm_${NAME}_stage1.log
SAVE_DIR=$REPO/products/task207/arm_${NAME}

echo "[$(date)] === Task #207 Phase 0: arm=${NAME}, loss_type=${LOSS_TYPE}, euclidean_qloss=${EUC_Q} ==="
echo "[$(date)] epochs=500, batch_size=1024, c=1.0, beta=0.5, sk_eps=[0,0,0]"

mkdir -p $SAVE_DIR
echo $$ > $SAVE_DIR/_TRAINING_PID

# 如果是 SLURM, CUDA_VISIBLE_DEVICES 已自动设置; 否则手动
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${1:-0}
fi

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task207_${NAME} \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 500 --batch_size 1024 \
    --loss_type $LOSS_TYPE \
    $(if [ "$EUC_Q" = "True" ]; then echo "--euclidean_qloss"; fi) \
    --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta 0.5 --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG 2>&1

rm -f $SAVE_DIR/_TRAINING_PID

echo "[$(date)] === Task #207 Phase 0 arm ${NAME} 完成 ==="
