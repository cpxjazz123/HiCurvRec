#!/bin/bash
# Task #303 / Issue #32 — DIAGNOSTIC inspect what T5 actually generates
# Purpose: See if T5 generates valid SIDs or garbage

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models

LOG_DIR=$REPO/logs/task303
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/diag_pred_inspect_${TS}.log

mkdir -p $LOG_DIR

echo "===== [DIAGNOSTIC PRED INSPECT] Task #303 Issue #32 ckpt predictions =====" | tee $LOG_FILE

BEST_CKPT=$(ls -t $REPO/products/task303/ckpt_hgrec_issue32/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
echo "best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

$PYTHON_BIN -c "
import sys, os, torch
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
    'code_path': '_t5_hrqvae_issue32_dual_axis_synergy.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config)
ckpt = torch.load('$BEST_CKPT', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.to(device)
model.eval()

valid_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len']
)
valid_dataloader = GenRecDataLoader(valid_dataset, batch_size=2, shuffle=False)

# Get first batch
batch = next(iter(valid_dataloader))
input_ids = batch['history'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['target'].to(device)

print(f'[INSPECT] input_ids shape: {input_ids.shape}')
print(f'[INSPECT] labels shape: {labels.shape}')
print(f'[INSPECT] label 0 (target): {labels[0].tolist()}')
print(f'[INSPECT] label 1 (target): {labels[1].tolist()}')

# Inspect item_to_code
print(f'[INSPECT] First 5 item_to_code entries:')
for i, (k, v) in enumerate(valid_dataset.item_to_code.items()):
    if i >= 5:
        break
    print(f'  item_id={k} -> {v.tolist() if hasattr(v, \"tolist\") else v}')

# Find which item_id corresponds to label[0]
print(f'[INSPECT] Reverse lookup for label[0]:')
target_label = tuple(labels[0].tolist())
for k, v in valid_dataset.item_to_code.items():
    if tuple(v.tolist()) == target_label:
        print(f'  Found: item_id={k} -> {v.tolist()}')
        break

# Generate predictions
with torch.no_grad():
    preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=20)

print(f'[INSPECT] preds raw shape: {preds.shape}')
print(f'[INSPECT] preds[0] (full sequence, beam=1): {preds[0].tolist()}')
print(f'[INSPECT] preds[0, 1:] (without start): {preds[0, 1:].tolist()}')

# Reshape
preds_clean = preds[:, 1:].reshape(input_ids.shape[0], 20, -1)
print(f'[INSPECT] preds_clean shape: {preds_clean.shape}')
print(f'[INSPECT] preds_clean[0, 0] (best beam): {preds_clean[0, 0].tolist()}')
print(f'[INSPECT] preds_clean[0, 1] (2nd beam): {preds_clean[0, 1].tolist()}')
print(f'[INSPECT] preds_clean[0, 19] (20th beam): {preds_clean[0, 19].tolist()}')

# Are predictions valid 4-token sequences?
print(f'[INSPECT] preds_clean[0] token ranges: min={preds_clean[0].min()}, max={preds_clean[0].max()}')
print(f'[INSPECT] labels[0] token ranges: min={labels[0].min()}, max={labels[0].max()}')
" 2>&1 | tee -a $LOG_FILE