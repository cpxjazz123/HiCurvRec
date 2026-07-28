#!/bin/bash
# Task #164 Phase B κ-decouple → Stage 3: T5-small 60M 训练 on Phase B SID
# 镜像 task160_hgrec_t5small60m_stage3.sh,但 code_path 用 phase_b SID
# R12: save_strategy + save_total_limit (由 task84_hgrec_stage3_train.py 处理)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task164_p1_s3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task164
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_60m_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task164/phase_b_kappa_decouple/${TS}/stage3_60m
mkdir -p $PROD_DIR

echo "===== [Task #164 Stage 3] Phase B SID + T5-small 60M launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-small 60M (6 enc + 6 dec, d_model=512, d_ff=2048, 8 heads d_kv=64) + code_default-like SID Phase B" | tee -a $LOG_FILE
echo "GPU 0 (CUDA_VISIBLE_DEVICES=0)" | tee -a $LOG_FILE
echo "Code path: _t5_rqvae_phase_b_kappa_decouple.npy" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs baseline target R@10 > 0.1051 (Task #88) or > 0.1058 (phonism)" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=0 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_phase_b_kappa_decouple.npy \
    --codebook_size 32 64 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 6 \
    --d_model 512 \
    --d_ff 2048 \
    --num_heads 8 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path $PROD_DIR \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 40 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #164 Stage 3] completed at $(date) =====" | tee -a $LOG_FILE
