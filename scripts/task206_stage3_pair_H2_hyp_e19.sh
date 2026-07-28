#!/bin/bash
# Task #206 Stage 3 — 双曲 ep19 (Pair H2)
#
# .npy 已存在 (Instruments_t5_rqvae_hyp_e19.npy, 双曲 ep19 collision 42.81%)
# 跟 Arm B 欧式 2000ep (43.30%) 配对:0.49pp 差距,单一变量 = 几何 (--loss_type + --euclidean_qloss)
#
# 用 task84_hgrec_stage3_train.py 跑 (论文 alignment + 早期停止 + ckpt 保存)
# GPU 0 (空闲), GPU 1 在跑别的, GPU 2 备用, GPU 3 在跑 euc_e9 stage2

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206 $REPO/products/task206/t5small_hyp_e19

CODE_PATH_NAME=Instruments_t5_rqvae_hyp_e19.npy
CODE_FULL_PATH=$REPO/HG-Rec/dataset/Instruments/$CODE_PATH_NAME
CODE_ARG=_t5_rqvae_hyp_e19.npy
LOG_FILE=$REPO/logs/task206/stage3_pair_H2_hyp_e19.log
GPU=0

# .npy 已存在 (前面 43% 对齐时已落盘)
echo "[$(date)] 双曲 ep19 .npy exists check: $CODE_FULL_PATH" | tee $LOG_FILE
if [ ! -f $CODE_FULL_PATH ]; then
    echo "❌ 双曲 ep19 .npy 不存在,先跑 Stage 2" | tee -a $LOG_FILE
    exit 1
fi
echo "[$(date)] ✓ $CODE_FULL_PATH ready" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_stage3_hyp_e19 \
python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --device cuda:0 \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path $CODE_ARG \
    --batch_size 256 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 \
    --d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64 \
    --dropout_rate 0.1 \
    --vocab_size 1025 \
    --pad_token_id 0 --eos_token_id 0 \
    --feed_forward_proj relu \
    --max_len 20 \
    --codebook_size 64 128 256 1 \
    --beam_size 20 \
    --seed 2025 \
    --log_path $REPO/logs/task206/ \
    --save_path $REPO/products/task206/t5small_hyp_e19/ \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task206/_STAGE3_PID_hyp_e19
echo "[$(date)] Stage 3 双曲 ep19 启动 PID=$TRAIN_PID GPU=$GPU, 跟 Arm B (欧式 43%) 同时跑" | tee -a $LOG_FILE
