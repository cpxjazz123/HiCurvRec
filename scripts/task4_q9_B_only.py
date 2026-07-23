#!/usr/bin/env python3
# task6_q9_B_only.py — 补跑 Group B 算法
# 量 9 — Linear probe from SID prefix → flan-t5 embedding

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import torch
import numpy as np

# Import the LinearProbe + train_probe from task16_q9
from task16_q9 import LinearProbe, train_probe, EMB_PATH

EMB = torch.load(EMB_PATH, weights_only=False)
N, D = EMB.shape
print(f'flan-t5 embedding: {EMB.shape}')

# Group B SID path
B_SID = '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-08/00-32-52/pickle/merged_predictions_tensor.pt'
sid = torch.load(B_SID, weights_only=False)
if sid.shape[0] == 4 and sid.shape[1] == 11924:
    sid = sid.T   # (11924, 4)
print(f'B SID shape: {sid.shape}')

# Per-l probe
cos_per_l = []
delta_per_l = []
prev = 0.0
for l in range(sid.shape[1] + 1):
    if l == 0:
        cos_l = 0.0
    else:
        prefix = sid[:, :l]
        print(f'  l={l}: training probe on prefix shape {prefix.shape}')
        cos_l, _ = train_probe(prefix, EMB, n_epochs=200)
    delta = cos_l - prev
    cos_per_l.append(cos_l)
    delta_per_l.append(delta)
    print(f'    l={l}: cos={cos_l:.4f}, Δ={delta:.4f}')
    prev = cos_l

result = {
    'algorithm': 'B_mmq',
    'L': sid.shape[1],
    'cos_per_l': cos_per_l,
    'delta_per_l': delta_per_l,
}

# Save (merge into existing all.json)
ALL_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task18/task6_q9_all.json'
if os.path.exists(ALL_PATH):
    with open(ALL_PATH) as f:
        all_results = json.load(f)
else:
    all_results = {}
all_results['B_mmq'] = result

with open(ALL_PATH, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\n=== Updated → {ALL_PATH} ===')

# Save individual
B_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task18/task6_q9_B_mmq.json'
with open(B_PATH, 'w') as f:
    json.dump(result, f, indent=2)
print(f'  → {B_PATH}')