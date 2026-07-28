#!/bin/bash
# Task #237 Stage 4 — Arm B partial Sinkhorn Stage 4 eval + slice
# Issue #10 Gate 1 Arm B (max_sinkhorn_iters=10) → T5-mini test eval

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task237_s4
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/logs/task237 $REPO/verdicts

CKPTS=$(find $REPO/products/task237/t5mini_armB/Instruments -name "HG_Rec_best.pth" 2>/dev/null | sort)
if [ -z "$CKPTS" ]; then
    echo "❌ No HG_Rec_best.pth in products/task237/t5mini_armB/Instruments/"
    exit 1
fi
BEST_CKPT=$(echo "$CKPTS" | tail -1)
CKPT_MTIME=$(stat -c '%y' "$BEST_CKPT")
CKPT_SIZE=$(stat -c '%s' "$BEST_CKPT")
echo "✅ Stage 3 best ckpt: $BEST_CKPT"
echo "   mtime=$CKPT_MTIME  size=$CKPT_SIZE bytes"

LOG_FILE=$REPO/logs/task237/stage4_armB_eval_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log
RESULT_JSON=$REPO/verdicts/task237_armB_test_metrics.json

echo "===== [Task #237 Stage 4] Test eval launched at $(date) =====" | tee $LOG_FILE
echo "ckpt: $BEST_CKPT" | tee -a $LOG_FILE

python3 -u <<PYTHON_EOF 2>&1 | tee -a "$LOG_FILE"
import sys, os, json, time
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
    'batch_size': 256,
    'infer_size': 96,
    'lr': 1e-4,
    'device': 'cuda:0',
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'dropout_rate': 0.1,
    'vocab_size': 1025,
    'pad_token_id': 0,
    'eos_token_id': 0,
    'feed_forward_proj': 'relu',
    'max_len': 20,
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_rqvae_task237_armB.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt_path = '$BEST_CKPT'
print(f'[Task #237 Stage 4] Loading best ckpt: {ckpt_path}', flush=True)
sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(sd)
model.to(device)
model.eval()

test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Task #237 Stage 4] Test dataset size: {len(test_dataset)}', flush=True)

t0 = time.time()
avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
elapsed = time.time() - t0
print(f'[Task #237 Stage 4] Eval time: {elapsed:.1f}s', flush=True)
print(f'[Task #237 Stage 4] Test recalls: {avg_recalls}', flush=True)
print(f'[Task #237 Stage 4] Test ndcgs: {avg_ndcgs}', flush=True)

result = {
    'task': 'task237_armB_partial_sinkhorn10',
    'sid_file': config['code_path'],
    'ckpt_path': ckpt_path,
    'ckpt_mtime': '$CKPT_MTIME',
    'ckpt_size': $CKPT_SIZE,
    'eval_time_sec': round(elapsed, 1),
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'baseline_comparison': {
        'HG-Rec #84': {'R@5': 0.0816, 'R@10': 0.1020, 'R@20': 0.1279},
        '#200 dual_v5 (truncated ep93)': {'R@5': 0.0756, 'R@10': 0.0915, 'R@20': 0.1119},
        '#233 dual_v5 RERUN (c=10 ep1)': {'R@5': None, 'R@10': None, 'R@20': None},
        '#237 Arm B partial sk10 (this)': {k: float(v) for k, v in avg_recalls.items()},
    },
}
os.makedirs(os.path.dirname('$RESULT_JSON'), exist_ok=True)
with open('$RESULT_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Task #237 Stage 4] Saved: $RESULT_JSON', flush=True)
PYTHON_EOF

echo "===== [Task #237 Stage 4] DONE at $(date) =====" | tee -a $LOG_FILE
echo "Result JSON: $RESULT_JSON"
