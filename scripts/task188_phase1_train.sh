#!/bin/bash
# Task #188 Phase 1 — Stage 1 RQ-VAE retraining with --save_limit 50
# 目的: 保留完整碰撞率谱系 (epoch 4-999 全留), 给 Phase 2 选 4 档 collision 用

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task188
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task188
CKPT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task188/hrqvae_save_limit50
mkdir -p $LOG_DIR $CKPT_DIR

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase1_train_${TS}.log

echo "===== [Task #188 Phase 1] Stage 1 retraining at $(date) =====" | tee $LOG_FILE
echo "Recipe: identical to Task #181 (β=1.0, Sinkhorn OFF, Phase 0.6 paper-aligned)" | tee -a $LOG_FILE
echo "Diff: --save_limit 50 (vs Task #181 default 5)" | tee -a $LOG_FILE
echo "GPU: $CUDA_VISIBLE_DEVICES, save_dir: $CKPT_DIR" | tee -a $LOG_FILE

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

python3 -u train_hrqvae.py \
    --lr 1e-3 \
    --epochs 1000 \
    --batch_size 1024 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --weight_decay 0.0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 1.0 \
    --layers 512 256 128 64 \
    --save_limit 50 \
    --ckpt_dir $CKPT_DIR \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #188 Phase 1] exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

# R12: 清理 PID 文件
rm -f /home/wlia0047/ar57/wenyu/GeneRec/products/task188/_PHASE1_PID

echo "✅ Task #188 Phase 1 Stage 1 retraining complete" | tee -a $LOG_FILE
echo "Final ckpt dir: $CKPT_DIR" | tee -a $LOG_FILE
echo "Best collision model saved. 全 epoch 快照保留 (save_limit=50)." | tee -a $LOG_FILE
