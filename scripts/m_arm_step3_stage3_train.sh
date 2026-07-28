#!/bin/bash
# 方向 F Stage 3: T5-mini 9.18M 训练 on v11/v12 SID (ep4 ckpts, low collision)
# 目标: 即使 5cond FAIL, test short-train low-collision SID 是否帮下游 R@10
# 对照: HG-Rec baseline (Task #84) R@10=0.1020 on Musical_Instruments

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_v11_s3
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/m_arm_step3
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')

PROD_DIR_V11=/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/step3_v11_angdim8_ep4_sid/${TS}
PROD_DIR_V12=/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/step3_v12_angdim16_ep4_sid/${TS}
mkdir -p $PROD_DIR_V11 $PROD_DIR_V12 $LOG_DIR

CODE_FILE_V11=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_m_arm_v11_angdim8_ep4.npy
CODE_FILE_V12=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_m_arm_v12_angdim16_ep4.npy

if [ ! -f "$CODE_FILE_V11" ] || [ ! -f "$CODE_FILE_V12" ]; then
    echo "❌ Missing v11/v12 SID .npy"
    exit 1
fi

# ── Run 1: v11 (angdim8, coll 5.45%) ────────────────────────
LOG_V11=$LOG_DIR/v11_s3_${TS}.log
echo "[$(date)] Launching v11 Stage 3 on GPU 0" | tee $LOG_V11
echo "  SID: $CODE_FILE_V11" | tee -a $LOG_V11
echo "  Output: $PROD_DIR_V11" | tee -a $LOG_V11

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_v11_s3 \
    nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_m_arm_v11_angdim8_ep4.npy \
    --codebook_size 64 128 256 1 \
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
    --save_path $PROD_DIR_V11 \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_V11 2>&1 &

PID_V11=$!
echo $PID_V11 > /home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/_TRAINING_V11_PID
echo "  v11 PID: $PID_V11" | tee -a $LOG_V11

# ── Run 2: v12 (angdim16, coll 8.35%) ────────────────────────
LOG_V12=$LOG_DIR/v12_s3_${TS}.log
echo "[$(date)] Launching v12 Stage 3 on GPU 1" | tee $LOG_V12
echo "  SID: $CODE_FILE_V12" | tee -a $LOG_V12
echo "  Output: $PROD_DIR_V12" | tee -a $LOG_V12

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_v12_s3 \
    mkdir -p /home/wlia0047/.triton/cache_marm_v12_s3
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_marm_v12_s3 \
    nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_m_arm_v12_angdim16_ep4.npy \
    --codebook_size 64 128 256 1 \
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
    --save_path $PROD_DIR_V12 \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_V12 2>&1 &

PID_V12=$!
echo $PID_V12 > /home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/_TRAINING_V12_PID
echo "  v12 PID: $PID_V12" | tee -a $LOG_V12
echo "[$(date)] Both v11/v12 Stage 3 launched." | tee -a $LOG_V11 $LOG_V12
