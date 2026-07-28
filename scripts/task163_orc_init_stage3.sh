#!/bin/bash
# Task #163 Stage 3 — HG-Rec T5-mini 5.5M + ORC-init free-curv SID
# Fork of task162 stage3, target = _t5_rqvae_orc_init.npy (vs #162 paper_free_curv)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2  # R7: GPU 2 完全空闲 (avoid GPU 1 #162 + GPU 0 #157)
export HF_HOME=/home/wlia0047/wenyu/scratch/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task163

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log

PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task163/_TRAINING_PID
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task163
echo $$ > $PID_FILE

echo "===== [Task #163 Stage 3] T5-mini 5.5M + ORC-init free-curv SID launched at $(date) =====" | tee $LOG_FILE
echo "code_path: _t5_rqvae_orc_init.npy" | tee -a $LOG_FILE
echo "GPU 2 (avoid collision with #162 GPU 1 + #157 GPU 0)" | tee -a $LOG_FILE
echo "vocab_size=4500 (R11.3 fix: item2code cumulative offset → max token=4478 < 4500)" | tee -a $LOG_FILE

export BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task163/ckpt_hgrec

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --batch_size 256 --num_epochs 200 --lr 1e-4 --device cuda \
    --num_layers 4 --num_decoder_layers 4 \
    --d_model 128 --d_ff 512 --num_heads 4 --d_kv 32 \
    --dropout_rate 0.1 --vocab_size 4500 \
    --max_len 20 --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --codebook_size 32 64 256 1 \
    --code_path _t5_rqvae_orc_init.npy \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task163/ckpt_hgrec \
    --early_stop 20 --topk_list 5 10 20 --beam_size 20 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #163 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE
echo "BEST_CKPT=${BEST_CKPT}" | tee -a $LOG_FILE
rm -f $PID_FILE