#!/bin/bash
# Task #187 — 4 层 encoder T5-mini training
# 单变量: --num_layers 4 (vs Task #181 6 层)
# 其他 recipe 完全跟 Task #181 一致

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task187
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task187
SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task187/t5small_4layer
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/train_${TS}.log
mkdir -p $LOG_DIR $SAVE_PATH

echo "===== [Task #187] 4-encoder training at $(date) =====" | tee $LOG_FILE
echo "Single variable: --num_layers 4 (vs Task #181 6)" | tee -a $LOG_FILE
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a $LOG_FILE
echo "Save path: $SAVE_PATH" | tee -a $LOG_FILE

python3 -u /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --batch_size 1024 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 1e-4 \
    --num_layers 4 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --dropout_rate 0.1 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --codebook_size 64 128 256 1 \
    --code_path _t5_rqvae_paper_fix.npy \
    --mode train \
    --save_path $SAVE_PATH \
    --log_path $LOG_DIR \
    --seed 42 \
    --early_stop 20 \
    --topk_list 5 10 20 \
    --beam_size 20 \
    2>&1 | tee -a $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #187] train exit code: $EXIT_CODE at $(date) =====" | tee -a $LOG_FILE
[ $EXIT_CODE -ne 0 ] && exit 1

rm -f /home/wlia0047/ar57/wenyu/GeneRec/products/task187/_TRAINING_PID
echo "✅ Task #187 4-encoder training complete" | tee -a $LOG_FILE

# === Stage 4 test eval ===
EVAL_TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
EVAL_LOG=$LOG_DIR/stage4_eval_${EVAL_TS}.log

echo "" | tee -a $EVAL_LOG
echo "===== [Task #187 Stage 4] test eval at $(date) =====" | tee $EVAL_LOG

BEST_CKPT=$(ls -t $SAVE_PATH/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "❌ No best ckpt found in $SAVE_PATH" | tee -a $EVAL_LOG
    exit 1
fi
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task187_encoder4_metrics.json
echo "Best ckpt: $BEST_CKPT" | tee -a $EVAL_LOG

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$EVAL_LOG"
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import torch
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
)
_s4_mod = _ilu.module_from_spec(_s4_spec)
_s4_spec.loader.exec_module(_s4_mod)
evaluate = _s4_mod.evaluate

config = {
    'batch_size': 256, 'infer_size': 96, 'lr': 1e-4, 'device': 'cuda:0',
    'num_layers': 4,                        # ← 单变量
    'num_decoder_layers': 4,
    'd_model': 128, 'd_ff': 1024,
    'num_heads': 6, 'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
    'feed_forward_proj': 'relu', 'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_rqvae_paper_fix.npy',
    'topk_list': [5, 10, 20], 'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
print(f'[Task #187 Stage 4] Loading best ckpt...', flush=True)
model.load_state_dict(torch.load('$BEST_CKPT', map_location='cpu'))
model.to(device)

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #187 Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task187_encoder4',
    'recipe': 'T5-mini + Phase 0.6 paper-aligned + num_layers=4 (vs Task #181 6)',
    'best_ckpt': '$BEST_CKPT',
    'selected_by': 'best_val_NDCG@20',
    'beam_size': 20,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #187 Stage 4] Saved: $RESULT_JSON', flush=True)
print(f'[Task #187 Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #187 Stage 4] Test NDCGs: {avg_ndcgs}', flush=True)
PYTHON_EOF

EVAL_EXIT=${PIPESTATUS[0]}
echo "===== [Task #187 Stage 4] exit code: $EVAL_EXIT at $(date) =====" | tee -a $EVAL_LOG
[ $EVAL_EXIT -ne 0 ] && exit 1

echo "✅ Task #187 full pipeline complete" | tee -a $EVAL_LOG
cat $RESULT_JSON | tee -a $EVAL_LOG
