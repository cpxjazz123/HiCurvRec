#!/bin/bash
# Task #200 Stage 4 — dual_v5 SID 评估 (R@10/NDCG)
#
# 用户清单 4 选项 B: 验证 v5 collision 84% SID 在 Stage 3 性能"持平".
# baseline:
#   - HG-Rec Task #84 test R@10=0.1020
#   - #181 Phase 0.6 test R@10=0.1057
#   - #188 val K0=64 mean R@10=0.1241
#
# 配置: T5-mini 9.18M, beam_size=20, GPU 0.
# 数据: dual_v5 SID (`Instruments_t5_rqvae_dual_v5.npy`)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p $REPO/logs/task200 $REPO/products/task200

DATA_DIR=$REPO/HG-Rec/dataset
SID_FILE="_t5_rqvae_dual_v5.npy"  # 跟 stage3 launcher 一致
GPU=0

# 找 best_collision_model.pth (R12 ckpt)
SAVE_DIR=$REPO/products/task200/t5mini_dual_v5
CKPT_PATTERN="$SAVE_DIR/*/best_*.pth"
LATEST_CKPT=$(ls -t $CKPT_PATTERN 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "❌ No Stage 3 ckpt found in $CKPT_PATTERN"
    echo "   必须先跑 task200_stage3_dual_v5.sh"
    exit 1
fi

LOG_FILE=$REPO/logs/task200/stage4_dual_v5_eval.log
rm -f $LOG_FILE

echo "[$(date)] === Stage 4 dual_v5 eval: ckpt=$LATEST_CKPT ===" | tee -a $LOG_FILE
echo "[$(date)] === baseline: HG-Rec #84 R@10=0.1020, #181 R@10=0.1057 ===" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU python3 -u $REPO/scripts/task174_v3_stage4_eval.py \
    --ckpt_path "$LATEST_CKPT" \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE \
    --codebook_size 64 128 256 1 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode evaluation \
    --beam_size 20 --infer_size 96 \
    > $LOG_FILE 2>&1 &
EVAL_PID=$!
echo $EVAL_PID > $REPO/products/task200/_EVAL_PID_stage4_v5
echo "[$(date)] Stage 4 dual_v5 eval launched PID=$EVAL_PID" | tee -a $LOG_FILE
echo "[$(date)] 监控: tail -f $LOG_FILE" | tee -a $LOG_FILE