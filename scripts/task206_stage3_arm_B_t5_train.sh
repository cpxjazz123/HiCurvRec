#!/bin/bash
# Task #206 Stage 3 — 欧式 Arm B T5-small 训练
#
# 跟 #84 baseline 完整对齐, 只换 --code_path.
# 用 task84_hgrec_stage3_train.py 跑 (已经有论文 alignment + 早期停止 + ckpt 保存).
#
# GPU 1 (Stage 2 刚结束用同卡, 想用 3 但 2000ep Stage 1 还在跑 → 选 1)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task206

# 等 Stage 2 .npy 落盘
CODE_PATH_NAME=Instruments_t5_rqvae_euclidean_v1.npy
CODE_FULL_PATH=$REPO/HG-Rec/dataset/Instruments/$CODE_PATH_NAME
CODE_ARG=_t5_rqvae_euclidean_v1.npy   # dataset.py: dataset_name + code_path → 必须只传 _t5_... 段
LOG_FILE=$REPO/logs/task206/stage3_arm_B.log
GPU=1

# 等 Stage 2 跑完 (检查文件 + 等进程退出)
echo "[$(date)] 等待 Stage 2 .npy 落盘..." | tee $LOG_FILE
while [ ! -f $CODE_FULL_PATH ]; do sleep 5; done
echo "[$(date)] ✓ Stage 2 .npy exists: $CODE_FULL_PATH" | tee -a $LOG_FILE

# 等进程结束 (避免 GPU 冲突)
STAGE2_PID=$(cat $REPO/products/task206/_STAGE2_PID_arm_B 2>/dev/null || echo 0)
if [ "$STAGE2_PID" != "0" ] && kill -0 $STAGE2_PID 2>/dev/null; then
    echo "[$(date)] 等待 Stage 2 PID=$STAGE2_PID 退出..." | tee -a $LOG_FILE
    wait $STAGE2_PID 2>/dev/null || true
fi
echo "[$(date)] Stage 2 done. 启动 Stage 3 Arm B." | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task206_stage3_arm_B \
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
    --save_path $REPO/products/task206/t5small_euclidean/ \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task206/_STAGE3_PID_arm_B
echo "[$(date)] Stage 3 Arm B launched PID=$TRAIN_PID" | tee -a $LOG_FILE