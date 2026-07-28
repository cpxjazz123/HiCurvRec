#!/usr/bin/env python3
"""Task #237 Phase 3 — Head/Body/Tail slice on Arm B (partial Sinkhorn) ckpt.

Compare against task209 phase3 slices (A0 baseline + A3 path_reg) — same Head/Body/Tail
definition (item popularity from train.parquet).
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import numpy as np
import pandas as pd
import torch
from collections import Counter

# === 1. 切片划分 (跟 task209 完全一致) ===
print("=== Phase 3 slice definition (Head 20% / Body 60% / Tail 20%) ===")
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
print(f"Total items: {n}, Head: {len(head_items)}, Body: {len(body_items)}, Tail: {len(tail_items)}")

# === 2. 评估 task237 Arm B ===
import importlib.util as _ilu
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
_s4 = _ilu.spec_from_file_location('s3_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s3 = _ilu.module_from_spec(_s4); _s4.loader.exec_module(_s3)
evaluate = _s3.evaluate

# 找 best ckpt
ckpt_dir = '/home/wlia0047/ar57/wenyu/GeneRec/products/task237/t5mini_armB/Instruments/'
candidates = sorted([
    os.path.join(ckpt_dir, d, 'HG_Rec_best.pth')
    for d in os.listdir(ckpt_dir)
    if os.path.exists(os.path.join(ckpt_dir, d, 'HG_Rec_best.pth'))
])
assert candidates, f"No HG_Rec_best.pth in {ckpt_dir}"
ckpt = candidates[-1]
print(f"Best ckpt: {ckpt}")

# SID + vocab (task237 用 4-digit dedup, vocab_size=11000)
sid_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy'
sid = np.load(sid_path)
print(f"SID shape: {sid.shape}, max digit: {sid.max()}, unique total: {len(np.unique(sid.reshape(-1, sid.shape[-1]), axis=0))}")
vocab_size = 1025  # ckpt trained with vocab=1025 (跟 task237 stage4 一致)  # +1 for max digit, +1 for pad
print(f"vocab_size={vocab_size}")

config = {
    'codebook_size': [64, 128, 256, 1],
    'num_layers': 6,
    'num_decoder_layers': 4,
    'd_model': 128,
    'd_ff': 1024,
    'num_heads': 6,
    'd_kv': 64,
    'vocab_size': vocab_size,  # 1025 for task237 ckpt
    'pad_token_id': 0,
    'eos_token_id': 0,
    'dropout_rate': 0.0,
    'feed_forward_proj': 'relu',
}
device = 'cuda:0'
model = HG_Rec(config).to(device)
sd = torch.load(ckpt, map_location=device, weights_only=False)
model.load_state_dict(sd)
model.eval()
print(f"✅ Loaded ckpt: {ckpt}")

test = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet')
results = {}
for slice_name, slice_items in [('head', head_items), ('body', body_items), ('tail', tail_items), ('all', None)]:
    if slice_items is not None:
        test_slice = test[test['target'].isin(slice_items)].reset_index(drop=True)
    else:
        test_slice = test
    if len(test_slice) == 0:
        print(f"  ⚠️ {slice_name}: 0 samples")
        results[slice_name] = {'n_samples': 0}
        continue
    if slice_items is not None:
        tmp_path = f'/home/wlia0047/ar57/wenyu/GeneRec/products/task237/phase3_test_{slice_name}.parquet'
        test_slice.to_parquet(tmp_path)
    else:
        tmp_path = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet'

    ds = GenRecDataset(
        dataset_path=tmp_path,
        code_path=sid_path,
        mode='evaluation',
        codebook_size=[64, 128, 256, 1],
        max_len=20,
    )
    dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
    recalls, ndcgs = evaluate(model, dl, [5, 10, 20], 20, device)
    results[slice_name] = {
        'n_samples': int(len(test_slice)),
        'R@5': float(recalls['Recall@5']),
        'R@10': float(recalls['Recall@10']),
        'R@20': float(recalls['Recall@20']),
        'N@5': float(ndcgs['NDCG@5']),
        'N@10': float(ndcgs['NDCG@10']),
        'N@20': float(ndcgs['NDCG@20']),
    }
    print(f"  {slice_name} (n={len(test_slice)}): R@10={recalls['Recall@10']:.4f} N@10={ndcgs['NDCG@10']:.4f}")
    if slice_items is not None and os.path.exists(tmp_path):
        os.remove(tmp_path)

with open('/home/wlia0047/ar57/wenyu/GeneRec/products/task237/phase3_slice_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 切片评估 saved to products/task237/phase3_slice_eval.json")
