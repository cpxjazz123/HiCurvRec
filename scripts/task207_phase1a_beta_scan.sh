#!/bin/bash
# Task #207 — Phase 1a: Euclidean β scan (3 臂 × 500 epoch)
# 探针: β ∈ {1.0, 2.0, 5.0} 能否把欧式 collision 从 43.8% 降到 ~9% (匹配双曲)
#
#SBATCH --job-name=task207_euc_beta_scan
#SBATCH --output=logs/task207/slurm_beta_%A_%a.log
#SBATCH --error=logs/task207/slurm_beta_%A_%a.log
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task207 $REPO/products/task207

DATA=$REPO/HG-Rec/dataset/Instruments/item_emb.parquet

BETAS=(1.0 2.0 5.0)
SLOT=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$SLOT]}

LOG=$REPO/logs/task207/euc_beta${BETA}_stage1.log
SAVE_DIR=$REPO/products/task207/euc_beta${BETA}

echo "[$(date)] === Task #207 Phase 1a: Euclidean β=${BETA} ==="
echo "[$(date)] loss_type=mse, euclidean_qloss=True, epochs=500, batch_size=1024"

mkdir -p $SAVE_DIR
echo $$ > $SAVE_DIR/_TRAINING_PID

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${1:-0}
fi

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task207_euc_b${BETA} \
python3 -u $REPO/HG-Rec/train_hrqvae.py \
    --data_path $DATA \
    --lr 1e-3 --epochs 500 --batch_size 1024 \
    --loss_type mse --euclidean_qloss \
    --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --beta $BETA --layers 512 256 128 64 \
    --curvatures 1.0,1.0,1.0 \
    --quant_loss_weight 1.0 \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $SAVE_DIR \
    > $LOG 2>&1

rm -f $SAVE_DIR/_TRAINING_PID
echo "[$(date)] === Task #207 Phase 1a Euclidean β=${BETA} 完成 ==="
