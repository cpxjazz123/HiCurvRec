#!/bin/bash
# Task #172 Stage 3 — T5-mini 9.18M + κ-Stereo + κ_max=4.0 SID

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task172_s3

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task172
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task172/t5mini_kappa_max4/${TS}
mkdir -p $PROD_DIR
mkdir -p $LOG_DIR

echo "===== [Task #172 Stage 3] T5-mini 9.18M + κ-Stereo + κ_max=4.0 launched at $(date) =====" | tee $LOG_FILE
echo "Code path: _t5_rqvae_phase_b_kappa_max4.npy" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "Target: test R@10 > 0.1058 (Stop hook 条件)" | tee -a $LOG_FILE

CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_max4.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

CUDA_VISIBLE_DEVICES=3 nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_phase_b_kappa_max4.npy \
    --codebook_size 32 64 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 256 \
    --d_ff 1024 \
    --num_heads 4 \
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
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task172/_TRAINING_PID
echo "PID: $TRAIN_PID" | tee -a $LOG_FILE

cat > /home/wlia0047/ar57/wenyu/GeneRec/products/task172/_STAGE4_TRIGGER.sh <<EOF2
#!/bin/bash
PID=\$(cat /home/wlia0047/ar57/wenyu/GeneRec/products/task172/_TRAINING_PID 2>/dev/null)
if [ -z "\$PID" ]; then
    echo "[\$(date)] ❌ _TRAINING_PID not found for #172" >> /home/wlia0047/ar57/wenyu/GeneRec/logs/task172/_stage4_trigger.log
    exit 1
fi
LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/task172/_stage4_trigger.log
echo "[\$(date)] Task #172 Stage 4 daemon watching PID=\$PID" >> \$LOG
while ps -p \$PID > /dev/null 2>&1; do
    sleep 60
done
echo "[\$(date)] Training PID \$PID 退出, 启动 Stage 4..." >> \$LOG
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task172_t5mini_kappa_max4_stage4_eval.sh
STAGE4_EXIT=\$?
echo "[\$(date)] Stage 4 exit: \$STAGE4_EXIT" >> \$LOG
EOF2
chmod +x /home/wlia0047/ar57/wenyu/GeneRec/products/task172/_STAGE4_TRIGGER.sh
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/products/task172/_STAGE4_TRIGGER.sh > /dev/null 2>&1 &
echo "Stage 4 daemon PID: $!" | tee -a $LOG_FILE