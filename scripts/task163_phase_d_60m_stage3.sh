#!/bin/bash
# Task #163 Phase D Stage 3 — T5-small 60M trained on Phase D ORC-init codebook
# Fork of task160_hgrec_t5small60m_stage3.sh with code_path=_t5_rqvae_orc_init.npy + vocab_size=5032

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task163

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase_d_60m_stage3_train_${TS}.log
mkdir -p $LOG_DIR

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task163/phase_d_60m/${TS}
mkdir -p $PROD_DIR
PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #163 Phase D + 60M Stage 3] ORC-init codebook + T5-small 60M launched at $(date) =====" | tee $LOG_FILE
echo "T5 config: 6 enc + 6 dec layers, d_model=512, d_ff=2048, 8 heads × d_kv=64 (~60M from-scratch)" | tee -a $LOG_FILE
echo "Codebook: Phase D ORC-init (best_loss ckpt @ ep 352, partial — Stage 1 NaN at ep 764)" | tee -a $LOG_FILE
echo "  L0 κ = +0.766, L1 κ = +1.350, L2 κ = +1.345 (vs Task #160 code_default L0-L2 all ≈ 0.5)" | tee -a $LOG_FILE
echo "  vocab_size = 5032 (Stage 2 output max token id = 5031 + 1, 跟 #162 同口径)" | tee -a $LOG_FILE
echo "GPU 2 (CUDA_VISIBLE_DEVICES=2, 完全空闲)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs Task #160 (code_default + T5-small 60M): R@10=0.0974 baseline" | tee -a $LOG_FILE
echo "vs Task #162 (paper_free_curv + T5-mini 5.5M): GPU 1 训练跑中" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=2 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_orc_init.npy \
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
    --vocab_size 5032 \
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
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #163 Phase D + 60M Stage 3] Training completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE