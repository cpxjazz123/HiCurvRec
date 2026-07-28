#!/bin/bash
# Task #168 Stage 3 — T5-5.5M + κ-Stereographic Phase B SID
# 2026-07-25 — 4th ablation capacity point (T5-5.5M, T5-mini 9.18M #166, T5-small 60M #165, T5-base 220M #167)
# T5-5.5M config: 5+5 layers, d_model=192, d_ff=1024, 3 heads × d_kv=64 (估算 ~5.5M params)
# 复用 Task #164 Phase B κ-Stereographic SID
# GPU 3 (R7: 完全空闲, 与 #165 GPU 0 / #166 Stage 4 GPU 1 / #167 GPU 2 并行不冲突)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task168_s3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task168
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task168/t5_5p5m_kappa_decouple/${TS}
mkdir -p $PROD_DIR
mkdir -p $LOG_DIR

echo "===== [Task #168 Stage 3] T5-5.5M + κ-Stereographic Phase B SID launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-5.5M (5 enc + 5 dec layers, d_model=192, d_ff=1024, 3 heads × d_kv=64) + κ-Stereographic Phase B SID" | tee -a $LOG_FILE
echo "GPU 3 (CUDA_VISIBLE_DEVICES=3, 完全空闲, 跟 #165 GPU 0 / #166 Stage 4 GPU 1 / #167 GPU 2 并行)" | tee -a $LOG_FILE
echo "Code path: _t5_rqvae_phase_b_kappa_decouple.npy (from #164)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs baseline target R@10 > 0.1058" | tee -a $LOG_FILE
echo "vs #161 T5-mini 9.18M 普通 SID R@10=0.1012 (capacity-matched control)" | tee -a $LOG_FILE
echo "vs #165 v3 κ-Stereographic + T5-small 60M 长训 (~6h)" | tee -a $LOG_FILE
echo "vs #166 κ-Stereographic + T5-mini 9.18M 长训 (~1.1h)" | tee -a $LOG_FILE
echo "vs #167 κ-Stereographic + T5-base 220M 长训 (~15h)" | tee -a $LOG_FILE

# Verify codebook exists
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_decouple.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

# 启动后台训练 (GPU 3, R7: 完全空闲)
CUDA_VISIBLE_DEVICES=3 nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_phase_b_kappa_decouple.npy \
    --codebook_size 32 64 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 5 \
    --num_decoder_layers 5 \
    --d_model 192 \
    --d_ff 1024 \
    --num_heads 3 \
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
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task168/_TRAINING_PID
echo "PID: $TRAIN_PID" | tee -a $LOG_FILE
echo "_TRAINING_PID written to products/task168/_TRAINING_PID" | tee -a $LOG_FILE

# R88 daemon: 训练结束后自动 Stage 4
cat > /home/wlia0047/ar57/wenyu/GeneRec/products/task168/_STAGE4_TRIGGER.sh <<EOF2
#!/bin/bash
PID=\$(cat /home/wlia0047/ar57/wenyu/GeneRec/products/task168/_TRAINING_PID 2>/dev/null)
if [ -z "\$PID" ]; then
    echo "[\$(date)] ❌ _TRAINING_PID not found" >> /home/wlia0047/ar57/wenyu/GeneRec/logs/task168/_stage4_trigger.log
    exit 1
fi
LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/task168/_stage4_trigger.log
echo "[\$(date)] Task #168 Stage 4 daemon watching PID=\$PID" >> \$LOG
while ps -p \$PID > /dev/null 2>&1; do
    sleep 60
done
echo "[\$(date)] Training PID \$PID 退出, 启动 Stage 4..." >> \$LOG
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task168_t5_5p5m_kappa_decouple_stage4_eval.sh
STAGE4_EXIT=\$?
echo "[\$(date)] Stage 4 exit: \$STAGE4_EXIT" >> \$LOG
EOF2
chmod +x /home/wlia0047/ar57/wenyu/GeneRec/products/task168/_STAGE4_TRIGGER.sh
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/products/task168/_STAGE4_TRIGGER.sh > /dev/null 2>&1 &
echo "Stage 4 daemon PID: $!" | tee -a $LOG_FILE