#!/usr/bin/env python3
"""
Task #209 Phase 3.2 — Head/Body/Tail 切片评估.

按 item 流行度 (train.parquet 中 item 出现次数) 分三档:
  Head: 前 20% (流行)
  Body: 中 60%
  Tail: 后 20% (冷门)

对 A0 / A3 ckpt 在 test set 上按切片报 Recall@10 / NDCG@10.
"""
import sys, os, json
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import numpy as np
import pandas as pd
import torch
from collections import Counter

# === 1. 切片划分 ===
print("=== Phase 3.2: Head/Body/Tail 切片划分 ===")
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

# 保存切片定义
slice_info = {
    'total': n,
    'head': sorted(head_items),
    'body': sorted(body_items),
    'tail': sorted(tail_items),
}
with open('/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase3_slices.json', 'w') as f:
    json.dump(slice_info, f)
print(f"✅ 切片 saved to phase3_slices.json")

# === 2. 按切片过滤 test set, 评估 A0/A3 ===
from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
import importlib.util as _ilu
_s4 = _ilu.spec_from_file_location('s3_fork', '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py')
_s3 = _ilu.module_from_spec(_s4); _s4.loader.exec_module(_s3)
evaluate = _s3.evaluate

ARMS = {
    'A0_baseline': {
        'ckpt': '/home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/Instruments/Jul-25-2026_16-29-42/HG_Rec_best.pth',
        'sid': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_paper_fix.npy',
        'codebook_size': [64, 128, 256, 1],
        'vocab_size': 1025,
    },
    'A3_path_reg': {
        'ckpt': None,  # Stage 3 训练完后填入
        'sid': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task209_A3.npy',
        'codebook_size': [64, 128, 256, 1],
        'vocab_size': 11000,  # 因 4-digit dedup 把 L3 推到 9072
    },
}

CODE_PATH_A3 = ARMS['A3_path_reg']['sid']
test = pd.read_parquet('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet')

# 找 A3 best ckpt
A3_CKPT_CANDIDATES = sorted([
    '/home/wlia0047/ar57/wenyu/GeneRec/products/task209/t5small_A3/Instruments/' + d + '/HG_Rec_best.pth'
    for d in os.listdir('/home/wlia0047/ar57/wenyu/GeneRec/products/task209/t5small_A3/Instruments/')
    if os.path.exists('/home/wlia0047/ar57/wenyu/GeneRec/products/task209/t5small_A3/Instruments/' + d + '/HG_Rec_best.pth')
])
if A3_CKPT_CANDIDATES:
    ARMS['A3_path_reg']['ckpt'] = A3_CKPT_CANDIDATES[-1]
    print(f"A3 ckpt: {ARMS['A3_path_reg']['ckpt']}")
else:
    print(f"⚠️ A3 ckpt 不存在 — A3 切片评估跳过, A0 单独跑")

results = {}
for arm_name, arm in ARMS.items():
    if arm['ckpt'] is None or not os.path.exists(arm['ckpt']):
        continue
    print(f"\n=== {arm_name} ===")
    # Build model (HG_Rec takes a config dict)
    config = {
        'codebook_size': arm['codebook_size'],
        'num_layers': 6,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'vocab_size': arm['vocab_size'],
        'pad_token_id': 0,
        'eos_token_id': 0,
        'dropout_rate': 0.0,
        'feed_forward_proj': 'relu',
    }
    model = HG_Rec(config).to('cuda:0')
    sd = torch.load(arm['ckpt'], map_location='cuda:0', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    print(f"  ✅ Loaded ckpt: {arm['ckpt']}")

    arm_results = {}
    for slice_name, slice_items in [('head', head_items), ('body', body_items), ('tail', tail_items)]:
        # 过滤 test set: target ∈ slice_items 的样本
        test_slice = test[test['target'].isin(slice_items)].reset_index(drop=True)
        if len(test_slice) == 0:
            print(f"  ⚠️ {slice_name}: 0 samples")
            arm_results[slice_name] = {'n_samples': 0}
            continue
        # 写临时 parquet
        tmp_path = f'/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase3_test_{slice_name}.parquet'
        test_slice.to_parquet(tmp_path)

        ds = GenRecDataset(
            dataset_path=tmp_path,
            code_path=arm['sid'],
            mode='evaluation',
            codebook_size=arm['codebook_size'],
            max_len=20,
        )
        dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
        recalls, ndcgs = evaluate(model, dl, [10], 20, 'cuda:0')
        n_samp = len(test_slice)
        arm_results[slice_name] = {
            'n_samples': int(n_samp),
            'R@10': float(recalls['Recall@10']),
            'N@10': float(ndcgs['NDCG@10']),
        }
        print(f"  {slice_name} (n={n_samp}): R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}")

        # 清理临时文件
        os.remove(tmp_path)
    results[arm_name] = arm_results

    # 同时算 overall (full test set)
    full_results = {}
    for slice_name, slice_items in [('all', None)]:
        ds = GenRecDataset(
            dataset_path='/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet',
            code_path=arm['sid'],
            mode='evaluation',
            codebook_size=arm['codebook_size'],
            max_len=20,
        )
        dl = GenRecDataLoader(ds, batch_size=128, shuffle=False)
        recalls, ndcgs = evaluate(model, dl, [10], 20, 'cuda:0')
        full_results = {
            'R@10': float(recalls['Recall@10']),
            'N@10': float(ndcgs['NDCG@10']),
        }
        print(f"  ALL: R@10={recalls['Recall@10']:.4f}  N@10={ndcgs['NDCG@10']:.4f}")
    results[arm_name]['all'] = full_results

with open('/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase3_slice_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 切片评估 saved to phase3_slice_eval.json")
