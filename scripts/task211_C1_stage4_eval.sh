#!/bin/bash
# Task #209 Phase 2c — A3 Stage 4 test eval
# 输入: products/task211/t5mini_C1/.../HG_Rec_best.pth (Stage 3 训练产出)
# 输出: verdicts/task211_C1_metrics.json
# GPU 2 (R7: A0/A3 分配到独立卡)
#
# 跟 #181 Stage 4 唯一区别: ckpt path 改 task211/t5mini_C1/

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task211_C1_stage4
mkdir -p $TRITON_CACHE_DIR

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task209
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase2c_A3_stage4_${TS}.log
mkdir -p $LOG_DIR

# Find best ckpt (R12: only one best_ckpt per arm)
BEST_CKPT=$(ls /home/wlia0047/ar57/wenyu/GeneRec/products/task211/t5mini_C1/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Best ckpt MISSING under products/task211/t5mini_C1/" | tee $LOG_FILE
    exit 1
fi
RESULT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task211_C1_metrics.json

echo "===== [Task #209 Phase 2c A3] Stage 4 test eval at $(date) =====" | tee $LOG_FILE
echo "Best ckpt: $BEST_CKPT" | tee -a $LOG_FILE
echo "Result JSON: $RESULT_JSON" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
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

# Vocab must match Stage 3 training (A3 = 11000 due to 4th-digit dedup spillover)
CODE_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task211_C1.npy'
CODEBOOK_SIZE = [64, 128, 256, 1]
VOCAB_SIZE = 11000
DEVICE = 'cuda:0'
MAX_LEN = 20
BEAM_SIZE = 20
TOPK_LIST = [5, 10, 20]
BEST_CKPT = '$BEST_CKPT'
RESULT_JSON = '$RESULT_JSON'

# Build model (HG_Rec takes a config dict)
_config = {
    'codebook_size': CODEBOOK_SIZE,
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'vocab_size': VOCAB_SIZE,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'dropout_rate': 0.0,
    'feed_forward_proj': 'relu',
}
model = HG_Rec(_config).to(DEVICE)

print(f"Loading best ckpt: {BEST_CKPT}")
sd = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(sd)
model.eval()
print("✅ Model loaded.")

# Load test dataset
test_dataset = GenRecDataset(
    dataset_path='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet',
    code_path=CODE_PATH,
    mode='evaluation',
    codebook_size=CODEBOOK_SIZE,
    max_len=MAX_LEN,
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate
print("Evaluating on test set...")
avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, TOPK_LIST, BEAM_SIZE, DEVICE)

metrics = {
    'task': 'task211_C1',
    'best_ckpt': BEST_CKPT,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}

# Compare with A0 (#181 baseline) for reference
print(f"\n=== A3 (Phase 2c) Test Results ===")
for k, v in avg_recalls.items():
    print(f"  {k}: {v:.4f}")
for k, v in avg_ndcgs.items():
    print(f"  {k}: {v:.4f}")

with open(RESULT_JSON, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n✅ Metrics saved to {RESULT_JSON}")
PYTHON_EOF
