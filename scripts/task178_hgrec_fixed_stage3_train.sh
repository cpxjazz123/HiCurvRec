#!/bin/bash
# Task #178 Stage 3 — T5-small 5.5M (跟 Task #84 baseline 配置完全一致) + 修正后 baseline SID
# 输出: products/task178/t5small_fixed_hgrec/<TS>/Instruments/<TS>/HG_Rec_best.pth

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task178_stage3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task178
TS=$(date -u +%b-%d-%Y_%H-%M-%S)
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
mkdir -p $LOG_DIR

echo "===== [Task #178 Stage 3] T5-small 5.5M + 修正后 baseline SID 启动 at $(date) =====" | tee $LOG_FILE

SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task178/t5small_fixed_hgrec/$TS
mkdir -p $SAVE_PATH

# Launch detached
nohup python3 scripts/task84_hgrec_stage3_train.py \
  --dataset_name Instruments \
  --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
  --code_path _t5_rqvae_fixed_hgrec.npy \
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
  --save_path $SAVE_PATH \
  --log_path $LOG_DIR \
  --seed 42 \
  --early_stop 20 \
  --beam_size 20 \
  --infer_size 96 \
  > $LOG_FILE 2>&1 &

STAGE3_PID=$!
echo $STAGE3_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task178/_STAGE3_PID
echo "Task #178 Stage 3 PID = $STAGE3_PID, save_path = $SAVE_PATH"
sleep 15
ps -p $STAGE3_PID -o pid,etime,cmd 2>&1 | head -3
echo "---log---"
tail -20 $LOG_FILE | sed -E 's/\x1b\[[0-9;]*[mK]//g' | grep -vE "^[[:space:]]*$" | tail -10