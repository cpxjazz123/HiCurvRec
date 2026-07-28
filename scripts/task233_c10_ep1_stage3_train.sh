#!/bin/bash
# Task #233 Stage 3 — T5-mini 9.18M (baseline 配置) + 方向 I c=10 ep1 SID
# 目的: 测 c=10 SID (51.41% collision) → T5 R@10. 预期 < 0.07 (vs baseline 0.1020).
# baseline 对照: HG-Rec Task #84 R@10=0.1020 (T5-mini + 64/128/256/dedup codebook, Musical_Instruments).
# R12: task84_hgrec_stage3_train.py 内置 save_strategy + save_total_limit.

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task233_s3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task233
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/c10_ep1_stage3_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/stage3_t5mini
mkdir -p $PROD_DIR

CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task233_c10_ep1.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee $LOG_FILE
    exit 1
fi

echo "===== [Task #233 Stage 3] c=10 ep1 SID + T5-mini 9.18M launched at $(date) =====" | tee $LOG_FILE
echo "Code path: $CODE_FILE" | tee -a $LOG_FILE
echo "Codebook size: [64, 128, 256, 1]" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs baseline target R@10 = 0.1020 (HG-Rec Task #84, T5-mini + Musical_Instruments)" | tee -a $LOG_FILE
echo "Recipe: product_manifold + angular_dim=2 + radial_dim=32 + c=10 + 50 ep train, SID 51.41% collision" | tee -a $LOG_FILE
echo "GPU 0" | tee -a $LOG_FILE
echo "Early stop patience: 20" | tee -a $LOG_FILE

# 启动后台 training (T5-mini 9.18M baseline 同配置, 200 epoch + early_stop 20)
CUDA_VISIBLE_DEVICES=0 nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task233_c10_ep1.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 256 \
    --d_ff 1024 \
    --num_heads 4 \
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
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_c10_jul-27-2026_16-18-09/_TRAINING_PID
echo "PID: $TRAIN_PID" | tee -a $LOG_FILE
echo "Tail log: tail -f $LOG_FILE" | tee -a $LOG_FILE
echo "===== [Task #233 Stage 3] launched at $(date), PID=$TRAIN_PID =====" | tee -a $LOG_FILE