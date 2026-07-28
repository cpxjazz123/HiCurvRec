#!/bin/bash
# Task #202 — Sinkhorn-on Stage 3 launcher (单臂 K0=64 baseline)
# 1 臂 T5-mini × 200 epoch, GPU 1
# 跟 #188 val K0=64 tier 直接 1:1 对比 Sinkhorn vs 默认 SID 的增量
# Sinkhorn SID: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k064_sk0.003.npy (K0=64, sk_eps=0.003)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task202

# Wait for Sinkhorn SID file (Stage 2.2) ready
SID_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k064_sk0.003.npy
echo "[$(date)] Waiting for Sinkhorn SID: $SID_FILE"
while [ ! -f "$SID_FILE" ]; do
    sleep 30
    if [ -f "$SID_FILE" ]; then
        echo "[$(date)] Sinkhorn SID ready"
        break
    fi
done

K0=64
GPU=1
SAVE_PATH=$REPO/products/task202/t5mini_k0${K0}
LOG_FILE=$REPO/logs/task202/stage3_k0${K0}.log
rm -f $LOG_FILE
echo "[$(date)] K0=$K0: Sinkhorn-on T5-mini training on cuda:$GPU (SID: _sk0.003.npy)" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_rqvae_k0${K0}_sk0.003.npy \
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
    --log_path $REPO/logs/task202/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    >> $LOG_FILE 2>&1

echo "==== Task #202 K0=$K0 Sinkhorn Stage 3 done at $(date) ====" | tee -a $REPO/logs/task202/stage3_done.log