#!/bin/bash
# Task #165 v3 Stage 3 长训 — 200 epoch + early_stop=30
# 复用 task164 已训 Stage 1 ckpt + κ-Stereographic Phase B SID
# 期望 test R@10 ≥ 0.1058 baseline (val R@10=0.1177 已有 +11.2%)
# R12: save_strategy + save_total_limit (由 task84_hgrec_stage3_train.py 处理)
# R88: 启动时 echo $TRAIN_PID > products/task165/_TRAINING_PID

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task165_v3_s3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task165
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/v3_stage3_60m_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task165/v3_long_stage3/${TS}/stage3_60m
mkdir -p $PROD_DIR

echo "===== [Task #165 v3 Stage 3] κ-Stereographic Phase B + T5-small 60M 长训 launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: T5-small 60M (6 enc + 6 dec, d_model=512, d_ff=2048, 8 heads d_kv=64) + Phase B κ-decouple SID" | tee -a $LOG_FILE
echo "GPU 0 (CUDA_VISIBLE_DEVICES=0)" | tee -a $LOG_FILE
echo "Code path: _t5_rqvae_phase_b_kappa_decouple.npy (from task164)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs baseline target R@10 > 0.1058 (vanilla+Sinkhorn)" | tee -a $LOG_FILE
echo "Early stop patience: 30 (vs v1: 20)" | tee -a $LOG_FILE

# 启动后台 training
CUDA_VISIBLE_DEVICES=0 nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_phase_b_kappa_decouple.npy \
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
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path $PROD_DIR \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 30 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task165/_TRAINING_PID
echo "PID: $TRAIN_PID" | tee -a $LOG_FILE
echo "_TRAINING_PID written to products/task165/_TRAINING_PID" | tee -a $LOG_FILE
echo "Tail log: tail -f $LOG_FILE" | tee -a $LOG_FILE
echo "===== [Task #165 v3 Stage 3] launched at $(date), PID=$TRAIN_PID =====" | tee -a $LOG_FILE

# 写循环守护 (R88): 等训练结束, 自动触发 Stage 4
cat > /home/wlia0047/ar57/wenyu/GeneRec/products/task165/_STAGE4_TRIGGER.sh <<EOF2
#!/bin/bash
# 训练结束后自动触发 Stage 4
while [ -f /home/wlia0047/ar57/wenyu/GeneRec/products/task165/_TRAINING_PID ]; do
    sleep 60
done
echo "[$(date)] Training PID 退出, 启动 Stage 4..." >> /home/wlia0047/ar57/wenyu/GeneRec/logs/task165/_stage4_trigger.log
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task165_v3_stage4_eval.sh
EOF2
chmod +x /home/wlia0047/ar57/wenyu/GeneRec/products/task165/_STAGE4_TRIGGER.sh
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/products/task165/_STAGE4_TRIGGER.sh > /dev/null 2>&1 &
echo "Stage 4 daemon PID: $!" | tee -a $LOG_FILE