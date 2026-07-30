#!/bin/bash
# Task #194 / Issue #40 Gate 1 — Stage 3 + Stage 4 (waits for GPU 1 free)
# Stage 1 done, Stage 2 done. This script waits for task327 Stage 3 (PID 1531589) to finish,
# then runs Stage 3 T5-mini + Stage 4 eval on freed GPU 1.
# Trigger: run after task327 Stage 3 finishes ~16:35 AEST.

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

LOG_DIR=$REPO/logs/task194_gate1
PRODUCTS_DIR=$REPO/products/task194/protocol_match_k064

# Wait for Stage 2 .npy to be created (Stage 2 takes ~5 min)
OUTPUT_NPY=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_protocol_match_k064.npy
echo "[$(date)] Waiting for Stage 2 .npy to appear..." | tee -a $LOG_DIR/main.log
WAIT_START=$(date +%s)
while [ ! -f "$OUTPUT_NPY" ]; do
    sleep 30
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [ $ELAPSED -gt 1800 ]; then
        echo "❌ Stage 2 .npy still missing after 30 min, abort" | tee -a $LOG_DIR/main.log
        exit 1
    fi
done
echo "✅ Stage 2 output: $OUTPUT_NPY ($(stat -c%s $OUTPUT_NPY) bytes)" | tee -a $LOG_DIR/main.log

mkdir -p $LOG_DIR

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task194_gate1_k064
mkdir -p $TRITON_CACHE_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# ---------------------------------------------------------------------------
# Wait for task327 Stage 3 (PID 1531589) to finish
# ---------------------------------------------------------------------------
TASK327_PID=1531589
echo "[$(date)] Waiting for task327 Stage 3 (PID $TASK327_PID) to finish..." | tee -a $LOG_DIR/main.log
while kill -0 $TASK327_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] task327 Stage 3 (PID $TASK327_PID) finished, GPU 1 should be free" | tee -a $LOG_DIR/main.log

# R7: Verify GPU 1 is actually free before launching
sleep 30
GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1 2>/dev/null | head -1)
echo "GPU 1 status: $GPU1_UTIL" | tee -a $LOG_DIR/main.log

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# ---------------------------------------------------------------------------
# Stage 3 — T5-mini training
# ---------------------------------------------------------------------------
STAGE3_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE3_LOG=$LOG_DIR/stage3_${STAGE3_TS}.log
STAGE3_SAVE=$PRODUCTS_DIR/t5mini_k064

echo "===== [Issue #40 Gate 1 / K=64 stage3_stage4] Stage 3 launched at $(date) =====" | tee "$STAGE3_LOG"
echo "Recipe: #84 baseline mirror (code_path=_t5_hrqvae_protocol_match_k064.npy)" | tee -a "$STAGE3_LOG"

python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_protocol_match_k064.npy \
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
    --save_path $STAGE3_SAVE \
    --log_path $LOG_DIR/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    >> $STAGE3_LOG 2>&1 &
STAGE3_PID=$!
echo "$STAGE3_PID" > $PRODUCTS_DIR/_STAGE3_PID
echo "Stage 3 PID: $STAGE3_PID" | tee -a "$STAGE3_LOG"

wait $STAGE3_PID
STAGE3_EXIT=$?
echo "===== Stage 3 exit: $STAGE3_EXIT at $(date) =====" | tee -a "$STAGE3_LOG"
if [ $STAGE3_EXIT -ne 0 ]; then
    echo "❌ Stage 3 failed" | tee -a "$STAGE3_LOG"
    exit 1
fi

STAGE3_BEST=$(ls -t $STAGE3_SAVE/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$STAGE3_BEST" ]; then
    echo "❌ Stage 3 best ckpt NOT FOUND" | tee -a "$STAGE3_LOG"
    exit 1
fi
echo "✅ Stage 3 best ckpt: $STAGE3_BEST" | tee -a "$STAGE3_LOG"

rm -f $PRODUCTS_DIR/_STAGE3_PID

# ---------------------------------------------------------------------------
# Stage 4 — Test eval
# ---------------------------------------------------------------------------
STAGE4_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
STAGE4_LOG=$LOG_DIR/stage4_${STAGE4_TS}.log

echo "===== [Issue #40 Gate 1 / K=64 stage3_stage4] Stage 4 launched at $(date) =====" | tee "$STAGE4_LOG"

python3 - <<PYEOF 2>&1 | tee -a "$STAGE4_LOG"
import sys, os, json, time
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/scripts')
import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location('task84_s3_train_fork', '$REPO/scripts/task84_hgrec_stage3_train.py')
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 6, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
    'num_heads': 6, 'd_kv': 64, 'vocab_size': 1025, 'max_len': 20,
    'pad_token_id': 0, 'eos_token_id': 0,
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_hrqvae_protocol_match_k064.npy',
    'dataset_name': 'Instruments',
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'mode': 'evaluation', 'beam_size': 20,
    'seed': 42,
}

data_loader = GenRecDataLoader(config)
test_data = data_loader.load_data()
test_ds = GenRecDataset(config, test_data)

model = HG_Rec(config['codebook_size'], config['num_layers'], config['num_decoder_layers'],
              config['d_model'], config['num_heads'], config['d_ff'], 1024,
              config['vocab_size'], config['max_len'], 0.1)
model = model.to(config['device'])

sd = torch.load('$STAGE3_BEST', map_location='cpu', weights_only=False)
if 'model_state_dict' in sd:
    sd = sd['model_state_dict']
model.load_state_dict(sd, strict=False)
model.eval()

t0 = time.time()
test_dl = GenRecDataLoader(test_ds, batch_size=config['infer_size'], shuffle=False)
metrics = evaluate(model, test_dl, [5, 10, 20], config['beam_size'], config['device'])
elapsed = time.time() - t0

result = {
    'task': 'task194_issue40_gate1_protocol_matched_k064',
    'recipe_mirror': '#84 baseline (4-element confusion-fixed, batch_size 1024, epochs 1000, NO Sinkhorn)',
    'ckpt': '$STAGE3_BEST',
    'beam_size': 20,
    'metrics': metrics,
    'elapsed_sec': round(elapsed, 2),
}
print(json.dumps(result, indent=2))
out_path = '$LOG_DIR/stage4_metrics.json'
with open(out_path, 'w') as f:
    json.dump(result, indent=2)
print(f"Saved to {out_path}")
PYEOF

STAGE4_EXIT=$?
echo "===== Stage 4 exit: $STAGE4_EXIT at $(date) =====" | tee -a "$STAGE4_LOG"
echo "✅ Issue #40 Gate 1 / K=64 protocol-matched control DONE" | tee -a "$STAGE4_LOG"
echo "Read result: cat $LOG_DIR/stage4_metrics.json" | tee -a "$STAGE4_LOG"