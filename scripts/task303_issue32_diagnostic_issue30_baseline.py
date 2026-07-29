#!/bin/bash
# Task #303 / Issue #32 — DIAGNOSTIC CONTROL eval with Issue #30 ckpt + Issue #30 SID
# Purpose: Verify eval pipeline reproduces Task #301 baseline R@10=0.1022

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models

LOG_DIR=$REPO/logs/task303
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/diag_issue30_baseline_${TS}.log

mkdir -p $LOG_DIR

echo "===== [DIAGNOSTIC BASELINE] Issue #30 ckpt + Issue #30 SID eval =====" | tee $LOG_FILE

ISSUE30_CKPT=$(ls -t $REPO/products/task301/ckpt_hgrec_issue30/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
echo "Issue #30 best ckpt: $ISSUE30_CKPT" | tee -a $LOG_FILE

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

$PYTHON_BIN -c "
import sys, os, torch, json
sys.path.insert(0, '$REPO/HG-Rec')
sys.path.insert(0, '$REPO/scripts')

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

import importlib.util as _ilu
_s4_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    '$REPO/scripts/task84_hgrec_stage3_train.py',
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
    'dataset_path': '$REPO/HG-Rec/dataset/',
    'codebook_size': [64, 128, 256, 1],
    'code_path': '_t5_hrqvae_issue30_per_layer_transforms.npy',  # *** Issue #30 SID (correct for Issue #30 ckpt) ***
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load('$ISSUE30_CKPT', map_location='cpu')
print(f'[BASELINE] ckpt keys: {len(ckpt)}')
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f'[BASELINE] missing={len(missing)}, unexpected={len(unexpected)}')
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
print(f'[BASELINE] Test dataset size: {len(test_dataset)}')

avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)

result = {
    'task': 'task301_issue30_BASELINE_repro',
    'note': 'CONTROL: Issue #30 ckpt + Issue #30 SID — expect R@10=0.1022 (Task #301 baseline)',
    'best_ckpt': '$ISSUE30_CKPT',
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
}
import json as _json
DIAG_JSON = '$REPO/verdicts/task301_issue30_baseline_repro.json'
os.makedirs(os.path.dirname(DIAG_JSON), exist_ok=True)
with open(DIAG_JSON, 'w') as f:
    _json.dump(result, f, indent=2)
print(f'[BASELINE] Saved: {DIAG_JSON}')
print(f'[BASELINE] Test recalls: {avg_recalls}')
print(f'[BASELINE] Test NDCGs: {avg_ndcgs}')
" 2>&1 | tee -a $LOG_FILE