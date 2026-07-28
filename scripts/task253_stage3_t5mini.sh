#!/bin/bash
# Task #253 Stage 3 — T5-mini 训练 50 epoch 用 Mobius residual SID
# 注意: 必须 cd HG-Rec/, 否则 data.dataset ModuleNotFoundError
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

CODE_PATH=_t5_rqvae_mobius_residual_issue13_gate2.npy
DATA=$REPO/HG-Rec/dataset/
LOG_DIR=$REPO/logs/task253
PROD_DIR=$REPO/products/task253/t5mini_50ep
mkdir -p $LOG_DIR $PROD_DIR

GPU=0
LOG_FILE=$LOG_DIR/stage3_train.out
rm -f $LOG_FILE
echo "[$(date)] === Task #253 Stage 3: T5-mini on Möbius residual SID, GPU=$GPU ===" | tee $LOG_FILE
echo "[$(date)] SID code path: $CODE_PATH (collision=0.0915, L0=73.4%, L1/L2=100%)" | tee -a $LOG_FILE
echo "[$(date)] R12: save_strategy=epoch + save_total_limit=1 (latest ckpt only)" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task253_stage3 \
python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $DATA \
    --code_path $CODE_PATH \
    --codebook_size 64 128 256 1 \
    --num_epochs 50 \
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
    --save_path $PROD_DIR \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 10 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $PROD_DIR/_TRAINING_PID
echo "[$(date)] Task #253 Stage 3 launched PID=$TRAIN_PID" | tee -a $LOG_FILE