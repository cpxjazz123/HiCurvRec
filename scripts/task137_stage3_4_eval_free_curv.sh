#!/bin/bash
# Task #137 Stage 3 + 4 — FreeCurvHRQVAE codebook (curv0.5) downstream
# Stage 3: HG-Rec T5-small training on _t5_hrqvae_poincare_curv0.5.npy codebook
# Stage 4: test eval → R@10 vs Task #84 baseline 0.1020 + Task #89 A-arm 0.1015
# R11.3: seed=42 (per [[user-no-multiseed-override]]); GPU 0/1 空闲 (R7)
# R11.4 dry-run: 写好脚本, 等 A-arm 完成后 launch

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# GPU 选择 (R7 — Task #137 retrain 占 GPU 1 → 用 GPU 0/2/3)
# Stage 3: 选 GPU 0 (Task #136 ETEGRec 占 GPU 2; GPU 3 空闲)
# 建议优先 GPU 3 (ETEGRec 训练长, GPU 2 占用; GPU 0 vs TIGER 残留保险)
GPU_ID=3
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task137
mkdir -p $LOG_DIR $PROD_DIR/ckpt_hgrec_curv0.5/Instruments

TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task137_stage3_train_curv0.5_${TS}.log
PID_FILE=$PROD_DIR/_STAGE3_TRAINING_PID

# Stage 3 code_path (Task #137 A-arm Stage 2 output)
CODE_SUFFIX=_t5_hrqvae_poincare_curv0.5.npy

echo "===== [Task #137 Stage 3 curv0.5] HG-Rec T5-small training launched at $(date) =====" | tee $LOG_FILE
echo "Dataset=Instruments code_path=${CODE_SUFFIX} (Task #137 A-arm Stage 2 output)" | tee -a $LOG_FILE
echo "GPU ${GPU_ID} (R7 — Task #137 retrain GPU 1, ETEGRec GPU 2)" | tee -a $LOG_FILE
echo "Stage 3 fork: scripts/task84_hgrec_stage3_train.py (R12 save_limit=1)" | tee -a $LOG_FILE

# Verify Stage 2 codebook exists
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments${CODE_SUFFIX}
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 not complete" | tee -a $LOG_FILE
    exit 1
fi

# R11.3: 默认 200 epoch + early_stop=20, beam=20 (与 Task #84 baseline 一致)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path ${CODE_SUFFIX} \
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
    --save_path $PROD_DIR/ckpt_hgrec_curv0.5/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #137 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #137 Stage 3] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

# R12 验证 ckpt 是否落盘
BEST_CKPT=$PROD_DIR/ckpt_hgrec_curv0.5/Instruments/*/HG_Rec_best.pth
if ls $BEST_CKPT 2>/dev/null; then
    echo "✅ R12 best ckpt 落盘: $(ls $BEST_CKPT | head -1)" | tee -a $LOG_FILE
    CKPT_PATH=$(ls $BEST_CKPT | head -1)
else
    echo "❌ R12 best ckpt MISSING" | tee -a $LOG_FILE
    exit 1
fi

rm -f "$PID_FILE"

# ===== Stage 4 — test eval =====
echo "" | tee -a $LOG_FILE
echo "===== [Task #137 Stage 4 curv0.5] test eval launched at $(date) =====" | tee -a $LOG_FILE

EVAL_LOG=$LOG_DIR/task137_stage4_eval_curv0.5_${TS}.log
EVAL_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task137_curv0_5_test_eval.json

# Use task136_tiger_test_eval.py as template (it imports evaluate + GenRecDataset properly)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task137_test_eval_free_curv.py \
    --ckpt_path "$CKPT_PATH" \
    --code_suffix ${CODE_SUFFIX} \
    --output_json ${EVAL_JSON} \
    --log_file ${EVAL_LOG} \
    2>&1 | tee -a "$EVAL_LOG"

echo "===== [Task #137 Stage 4 curv0.5] eval done at $(date) =====" | tee -a $LOG_FILE