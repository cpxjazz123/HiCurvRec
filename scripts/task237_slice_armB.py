#!/usr/bin/env python3
"""Task #237 Arm B Phase 3 — Head/Body/Tail 切片评估 (单 arm)."""
import sys, os, json, time
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import numpy as np
import pandas as pd
import torch
from collections import Counter

print("=== Task #237 Arm B Phase 3: Head/Body/Tail 切片划分 ===")
df = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet')
cnt = Counter()
for h in df['history']:
    for it in h:
        cnt[int(it)] += 1
for t in df['target']:
    cnt[int(t)] += 1

n = len(cnt)
sorted_items = sorted(cnt.items(), key=lambda x: -x[1])
head_cutoff = int(n * 0.2)
tail_cutoff = int(n * 0.8)
head_items = set([it for it, _ in sorted_items[:head_cutoff]])
tail_items = set([it for it, _ in sorted_items[tail_cutoff:]])
body_items = set([it for it, _ in sorted_items[head_cutoff:tail_cutoff]])

print(f"Total items: {n}")
print(f"Head: {len(head_items)}, Body: {len(body_items)}, Tail: {len(tail_items)}")

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu
_s4 = _ilu.spec_from_file_location('s3_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s3 = _ilu.module_from_spec(_s4); _s4.loader.exec_module(_s3)
evaluate = _s3.evaluate

ARM = {
    'name': 'Arm_B_partial_Sinkhorn10',
    'ckpt': '/home/wlia0047/ar57/wenyu/GeneRec/products/task237/t5mini_armB/Instruments/Jul-29-2026_00-39-46/HG_Rec_best.pth',
    'sid': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy',
    'codebook_size': [64, 128, 256, 1],
    'vocab_size': 1025,  # partial Sinkhorn = 0 collisions
}

config = {
    'codebook_size': ARM['codebook_size'],
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'vocab_size': ARM['vocab_size'],
    'pad_token_id': 0,
    'eos_token_id': 0,
    'dropout_rate': 0.0,
    'feed_forward_proj': 'relu',
    'dataset_name': 'Instruments',
    'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
    'code_path': '_t5_rqvae_task237_armB.npy',
    'max_len': 20,
}

device = torch.device('cuda:0')
model = HG_Rec(config).to(device)
sd = torch.load(ARM['ckpt'], map_location=device, weights_only=False)
model.load_state_dict(sd)
model.eval()
print(f"✅ Loaded: {ARM['ckpt']}")

test = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet')

results = {'task': 'task237_armB_slice', 'ckpt': ARM['ckpt'], 'sid': ARM['sid'], 'slices': {}}

for slice_name, slice_items in [('head', head_items), ('body', body_items), ('tail', tail_items)]:
    test_slice = test[test['target'].isin(slice_items)].reset_index(drop=True)
    if len(test_slice) == 0:
        print(f"  ⚠️ {slice_name}: 0 samples")
        results['slices'][slice_name] = {'n_samples': 0}
        continue
    tmp_path = f'/home/wlia0047/ar57/wenyu/GeneRec/products/task237/phase3_test_{slice_name}.parquet'
    test_slice.to_parquet(tmp_path)
    ds = GenRecDataset(
        dataset_path=tmp_path,
        code_path=ARM['sid'],
        mode='evaluation',
        codebook_size=ARM['codebook_size'],
        max_len=20,
    )
    dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
    recalls, ndcgs = evaluate(model, dl, [5, 10, 20], 20, device)
    n_samp = len(test_slice)
    results['slices'][slice_name] = {
        'n_samples': int(n_samp),
        'R@5': float(recalls['Recall@5']),
        'R@10': float(recalls['Recall@10']),
        'R@20': float(recalls['Recall@20']),
        'N@5': float(ndcgs['NDCG@5']),
        'N@10': float(ndcgs['NDCG@10']),
        'N@20': float(ndcgs['NDCG@20']),
    }
    print(f"  {slice_name} (n={n_samp}): R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}")
    os.remove(tmp_path)

# Also ALL
ds = GenRecDataset(
    dataset_path='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet',
    code_path=ARM['sid'],
    mode='evaluation',
    codebook_size=ARM['codebook_size'],
    max_len=20,
)
dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
recalls, ndcgs = evaluate(model, dl, [5, 10, 20], 20, device)
results['slices']['all'] = {
    'n_samples': len(test),
    'R@5': float(recalls['Recall@5']),
    'R@10': float(recalls['Recall@10']),
    'R@20': float(recalls['Recall@20']),
    'N@5': float(ndcgs['NDCG@5']),
    'N@10': float(ndcgs['NDCG@10']),
    'N@20': float(ndcgs['NDCG@20']),
}
print(f"  ALL: R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}")

out_path = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task237_armB_slice.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved: {out_path}")
