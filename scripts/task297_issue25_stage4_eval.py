#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #297 / Issue #25 Stage 4 — Test evaluation on Gate 3 ckpt

评估 task297 Gate 3 训练 ckpt (epoch 0/1 早停, R12 已落盘).
- best_ckpt: products/task297/t5mini_issue25_gate3/Instruments/Jul-29-2026_22-35-28/HG_Rec_best.pth
- code_path: _t5_hrqvae_issue25_gate2_k0128.npy
- 输出: Recall@5/10/20, NDCG@5/10/20 → 跟 HG-Rec baseline 0.1020 对比
"""
import glob
import json
import os
import sys

import torch

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')
sys.path.insert(0, REPO)

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu

# Reuse Stage 3 evaluate function
_s3_spec = _ilu.spec_from_file_location(
    'task84_s3_train_fork',
    f'{REPO}/scripts/task84_hgrec_stage3_train.py',
)
_s3_mod = _ilu.module_from_spec(_s3_spec)
_s3_spec.loader.exec_module(_s3_mod)
evaluate = _s3_mod.evaluate


# Config (跟 Gate 3 launch 一致)
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
    'dataset_path': f'{REPO}/HG-Rec/dataset/',
    'codebook_size': [128, 128, 256, 1],  # K=128 起点
    'code_path': '_t5_hrqvae_issue25_gate2_k0128.npy',
    'topk_list': [5, 10, 20],
    'beam_size': 20,
}

# Load best ckpt
ckpt_pattern = f'{REPO}/products/task297/t5mini_issue25_gate3/Instruments/*/HG_Rec_best.pth'
candidates = sorted(glob.glob(ckpt_pattern))
if not candidates:
    raise FileNotFoundError(f'❌ No HG_Rec_best.pth in {ckpt_pattern}')
best_ckpt = candidates[-1]
print(f'[Stage 4] Loading best ckpt: {best_ckpt}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = HG_Rec(config)
model.load_state_dict(torch.load(best_ckpt, map_location='cpu', weights_only=False))
model.to(device)
model.eval()

# Test dataset
test_dataset = GenRecDataset(
    dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
    code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
    mode='evaluation',
    codebook_size=config['codebook_size'],
    max_len=config['max_len'],
)
test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
print(f'[Stage 4] Test dataset size: {len(test_dataset)}')

# Evaluate
avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
print(f'[Stage 4] Test recalls: {avg_recalls}')
print(f'[Stage 4] Test ndcgs: {avg_ndcgs}')

# JSON output
result = {
    'best_ckpt': best_ckpt,
    'test_recalls': avg_recalls,
    'test_ndcgs': avg_ndcgs,
    'config_codebook_size': config['codebook_size'],
    'config_code_path': config['code_path'],
}
result_json = f'{REPO}/verdicts/task297_issue25_gate3_test_metrics.json'
os.makedirs(os.path.dirname(result_json), exist_ok=True)
with open(result_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[Stage 4] Test metrics saved: {result_json}')

# 跟 HG-Rec baseline 对比
HG_REC_BASELINE_R10 = 0.1020
r10 = avg_recalls.get('Recall@10', 0.0) if isinstance(avg_recalls, dict) else (
    avg_recalls[config['topk_list'].index(10)] if isinstance(avg_recalls, list) else 0.0
)
print(f'\n[Gate 3 决策] R@10 = {r10:.4f}, HG-Rec baseline = {HG_REC_BASELINE_R10}, '
      f'PASS={r10 > HG_REC_BASELINE_R10}')
