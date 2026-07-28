#!/bin/bash
# Task #136 Phase 2 — TIGER (Math ≈ HG-Rec c555, vanilla flavor)
# R11.3 自决: 复用 task84 c555 Stage 1 RQ-VAE ckpt (κ=0.5 接近欧氏 ≈ vanilla TIGER),
#              复用 In_stage3 fork task84_hgrec_stage3_train.py,
#              重训 seed=2025 (vs Task #84 用 seed=42, paper DECOR default)
# GPU 0 (R7 强制空闲, 4 GPU 同时跑 baseline)
# R12: ckpt save_limit=1 (task84 fork 已经实现)
#
# 数学等价的近似理由:
#   HG-Rec c=0.5 (κ=0.5) expmap0 在单位 ball 内变换, gradient flow 接近欧氏
#   (因为 Poincaré ball 在 |x|≪1/√κ ≈ 1.41 范围几乎是 Euclidean geodesic)
#   对照 vanilla TIGER (sentence-t5 + RQ-VAE 无 hyperbolicity) 结果 paper R@10=0.0574

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/task136_tiger_seed2025_${TS}.log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task136/_TIGER_PID

mkdir -p $LOG_DIR /home/wlia0047/ar57/wenyu/GeneRec/products/task136

echo "===== [Task #136 Phase 2 TIGER] launch at $(date) =====" | tee $LOG_FILE
echo "Dataset=Instruments code_path=_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy (vanilla-TIGER analog)" | tee -a $LOG_FILE
echo "GPU 0 (R7)" | tee -a $LOG_FILE
echo "Seed=2025 (DECOR paper default)" | tee -a $LOG_FILE

CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 ckpt absent" | tee -a $LOG_FILE
    exit 1
fi

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _curv_0.5_0.5_0.5_t5_hrqvae_poincare.npy \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task136/ckpt_tiger_seed2025/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 2025 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #136 TIGER] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #136 TIGER] Training completed at $(date), exit code: $EXIT_CODE =====" | tee -a $LOG_FILE

if ls /home/wlia0047/ar57/wenyu/GeneRec/products/task136/ckpt_tiger_seed2025/Instruments/*/HG_Rec_best.pth 2>/dev/null; then
    echo "✅ R12 best ckpt saved" | tee -a $LOG_FILE
else
    echo "❌ R12 best ckpt MISSING — Task #136 TIGER FAILED" | tee -a $LOG_FILE
    exit 1
fi

rm -f "$PID_FILE"
