#!/bin/bash
# Task #200 Stage 3 — dual_v5 SID 训练 (v5 ckpt Stage 2 推断的 SID)
#
# 用户清单 4 选项 B: 验证 v5 (collision 84%) 的 Stage 3 性能"持平"用户原话兜底.
# baseline (HG-Rec) test R@10=0.1020, #181 (默认 SID) test R@10=0.1057.
# 期望: val R@10 ≈ 0.10 (持平或略掉).
#
# 配置: T5-mini 9.18M, 200 epoch, early_stop=20, seed=42, beam_size=20.
# 数据: dual_v5 SID (`Instruments_t5_rqvae_dual_v5.npy`, 由 task200_stage2_codebook.py 推断).

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task200 $REPO/products/task200

DATA_DIR=$REPO/HG-Rec/dataset
SID_FILE="_t5_rqvae_dual_v5.npy"  # task84 拼接 = dataset_path + dataset_name + "/" + dataset_name + code_path
GPU=0
SAVE_DIR=$REPO/products/task200/t5mini_dual_v5
LOG_FILE=$REPO/logs/task200/stage3_dual_v5.log
rm -f $LOG_FILE

# 检查 SID file 存在 (上游拼接 = dataset_path + dataset_name + "/" + dataset_name + code_path,
# 但 launcher 内只检查 stage 2 推断产物, 用完整文件名校验)
SID_ACTUAL="Instruments${SID_FILE}"
if [ ! -f "$DATA_DIR/Instruments/$SID_ACTUAL" ]; then
    echo "❌ SID file not found: $DATA_DIR/Instruments/$SID_ACTUAL"
    echo "   必须先跑 task200_stage2_codebook.py (dual_v5 SID 推断)"
    exit 1
fi

echo "[$(date)] === Stage 3 dual_v5: T5-mini 9.18M, 200 epoch, early_stop=20, GPU=$GPU ===" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 --batch_size 256 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode train \
    --save_path $SAVE_DIR \
    --log_path $REPO/logs/task200/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
    > $LOG_FILE 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $REPO/products/task200/_TRAINING_PID_stage3_v5
echo "[$(date)] Stage 3 dual_v5 launched PID=$TRAIN_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE