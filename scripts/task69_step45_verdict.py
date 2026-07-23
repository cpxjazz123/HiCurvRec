#!/usr/bin/env python3
"""Task #69 Step 4-5: Collision rate + final verdict.

Step 4: 算 TIGER collision rate as independent signal — 评估即使 SID 唯一性高,
       实际下游 TIGER 推断时 prediction tensor 中的 SID 是否也会碰撞.

Step 5: 比较 global vs conditional, 写最终 verdict 文件.
"""
import os, json
import numpy as np
import torch
from collections import Counter

SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/products/task126_phonism_rqvae/semantic_ids.pt'
PRED_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/products/task87_tiger_baseline/stage4_tiger_infer/merged_predictions_dedup.pt'
BASELINE_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step1_global/baseline.json'
COND_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step23_conditional/result.json'
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step45_verdict'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task #69 Step 4-5: Collision rate + Final verdict')
print('=' * 70)

# Load SID
sid_dict = torch.load(SID_PATH, weights_only=False)
sid = sid_dict['semantic_ids'].numpy()
n_items, n_layers = sid.shape
K = 256

# --- Collision analysis on catalog SID ---
print('\n[Step 4a] Catalog SID collision analysis')
unique_3 = len(np.unique(sid, axis=0))
print(f'  Unique 3-token SIDs: {unique_3}/{n_items} ({unique_3/n_items*100:.2f}%)')
# Per-layer: count unique values
for l in range(n_layers):
    u = len(np.unique(sid[:, l]))
    counts = Counter(sid[:, l].tolist())
    most_common = counts.most_common(5)
    print(f'  Layer {l}: unique={u}/{K}, top-5 codes: {most_common}')

# --- TIGER prediction tensor collision ---
# Check if user predictions have intra-L1-bucket L2/L3 collisions
print('\n[Step 4b] TIGER prediction tensor L1-bucket collision')
try:
    pred = torch.load(PRED_PATH, weights_only=False)
    if isinstance(pred, dict):
        for k in pred:
            print(f'  dict key={k} type={type(pred[k]).__name__}')
            if hasattr(pred[k], 'shape'):
                print(f'    shape={pred[k].shape}')
    else:
        print(f'  Pred shape: {pred.shape}, dtype: {pred.dtype}')
        if pred.dim() == 3:
            # (n_users, topK, n_layers)
            n_users, topK, _ = pred.shape
            print(f'  n_users={n_users}, topK={topK}')
            # For each user's top-10, check L1 bucket distribution
            l1_per_user = pred[:, :, 0]  # (n_users, topK)
            # How many unique L1 buckets in each user's top-K?
            unique_l1_per_user = (l1_per_user.max(dim=1).values - l1_per_user.min(dim=1).values + 1)  # rough range
            # More accurately: count unique L1 values
            unique_l1_counts = []
            for u in range(min(n_users, 1000)):
                unique_l1_counts.append(len(set(l1_per_user[u].tolist())))
            mean_unique_l1 = float(np.mean(unique_l1_counts))
            print(f'  Mean unique L1 codes in user top-{topK}: {mean_unique_l1:.2f} / {topK}')
            print(f'  => Users see {mean_unique_l1/topK*100:.1f}% L1 diversity in top-{topK}')
except Exception as e:
    print(f'  Could not load pred: {e}')

# --- Step 5: Final verdict ---
baseline = json.load(open(BASELINE_PATH))
cond = json.load(open(COND_PATH))

verdict = {
    'task': 'task69_phonism_conditional_diagnosis',
    'data_source': 'Task #126 phonism RQ-VAE on sentence-t5-base 768d embeddings',
    'data_summary': {
        'n_items': int(n_items),
        'coverage': {
            'L0': f'{len(np.unique(sid[:, 0]))}/{K} ({len(np.unique(sid[:, 0]))/K*100:.1f}%)',
            'L1': f'{len(np.unique(sid[:, 1]))}/{K} ({len(np.unique(sid[:, 1]))/K*100:.1f}%)',
            'L2': f'{len(np.unique(sid[:, 2]))}/{K} ({len(np.unique(sid[:, 2]))/K*100:.1f}%)',
        },
        'unique_3token_sids': f'{unique_3}/{n_items} ({unique_3/n_items*100:.2f}%)',
        'l1_bucket_distribution': cond['bucket_size_distribution'],
    },
    'global_results': baseline['mantel_global'],
    'conditional_results': cond['conditional_results'],
    'global_vs_conditional': cond['comparison_global_vs_conditional'],
    'conclusions': [
        f"Dilution hypothesis for L2: PARTIALLY CONFIRMED. Conditional Mantel ρ (0.131) > global (0.085), Δ=+0.046 (54% relative gain).",
        f"Dilution hypothesis for L3: REJECTED. Conditional Mantel ρ (0.039) < global (0.079), L3 digits are NOISE within L1 buckets.",
        f"kNN L2 hit rate gain is small: 6.37% → 8.65% (+2.28pp). Absolute value still very low.",
        f"kNN L3 hit rate essentially zero in both: 0.34% (global) vs 0.56% (conditional).",
        f"L1 layer is under-utilized (69/256 codes) but combined 3-token SID uniqueness is 97.36% — items do get unique IDs.",
        f"The phonism RQ-VAE successfully produces 97.36% unique 3-token SIDs, but L3 layer does NOT preserve local topology. This is a 'silent failure' masked by high uniqueness rate.",
    ],
    'implications': [
        "If downstream model relies on L3 to distinguish fine-grained item similarity, it sees essentially random codes.",
        "The high unique-SID rate (97.36%) is necessary but not sufficient — codebook must also have local topology.",
        "Future work: investigate why L3 Mantel ρ DECREASES when conditioning on L1 — possible cause: L3 quantization doesn't capture residual variance meaningfully after L2.",
    ],
}
out_path = os.path.join(OUT_DIR, 'verdict.json')
with open(out_path, 'w') as f:
    json.dump(verdict, f, indent=2)
print(f'\nSaved verdict → {out_path}')
print('\n=== Final Summary ===')
for c in verdict['conclusions']:
    print(f'• {c}')
