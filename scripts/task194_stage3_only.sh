#!/bin/bash
# Task #194 — Stage 3 only launcher (Stage 1/2 done, SID files at HG-Rec/dataset/Instruments/)
# 4 臂 T5-mini × 200 epoch 并发, 4 GPU 各 1 臂
# Stage 4 + diagnose + verdict 在 chain dispatch 后续自动 fire

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task194

train_stage3() {
    local K0=$1
    local GPU=$2
    local SAVE_PATH=$REPO/products/task194/t5mini_k0${K0}
    local LOG_FILE=$REPO/logs/task194/stage3_k0${K0}.log
    rm -f $LOG_FILE
    echo "[$(date)] K0=$K0: T5-mini training on cuda:$GPU" | tee -a $LOG_FILE
    CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
        --dataset_name Instruments \
        --dataset_path $REPO/HG-Rec/dataset/ \
        --code_path _t5_rqvae_k0${K0}.npy \
        --codebook_size ${K0} 128 256 1 \
        --num_epochs 200 \
        --batch_size 256 \
        --lr 1e-4 \
        --num_layers 6 \
        --num_decoder_layers 4 \
        --d_model 128 \
        --d_ff 1024 \
        --num_heads 6 \
        --d_kv 64 \
        --vocab_size 1025 \
        --max_len 20 \
        --pad_token_id 0 \
        --eos_token_id 0 \
        --device cuda:0 \
        --mode train \
        --save_path $SAVE_PATH \
        --log_path $REPO/logs/task194/ \
        --seed 42 \
        --early_stop 20 \
        --beam_size 20 \
        --infer_size 96 \
        >> $LOG_FILE 2>&1
}

# 4 臂并发
train_stage3 32 0 &
train_stage3 64 1 &
train_stage3 128 2 &
train_stage3 256 3 &
wait

echo "==== Task #194 Stage 3 done at $(date) ====" | tee -a $REPO/logs/task194/stage3_done.log