#!/usr/bin/env python3
"""Task #69 Step 2-3: Conditional Mantel + kNN within L1 buckets (Version B).

按 L1 codeword 分桶, 在桶内 (size>=5) 跑 Mantel + kNN hit rate.
与 Step 1 global 对比, 判定稀释问题是否真的存在.

Output:
- /home/wlia0047/ar57/wenyu/GeneRec/result/task69_step23_conditional/result.json
"""
import os, json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/products/task126_phonism_rqvae/semantic_ids.pt'
EMB_PATH = '/home/wlia0047/ar57/wenyu/genrec/dataset/amazon/processed/toys/item_emb_sentence-t5-base.parquet'
BASELINE_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step1_global/baseline.json'
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step23_conditional'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task #69 Step 2-3: Conditional Mantel + kNN within L1 buckets')
print('=' * 70)

# Load SID + embedding
sid_dict = torch.load(SID_PATH, weights_only=False)
sid = sid_dict['semantic_ids'].numpy()
n_items, n_layers = sid.shape

emb = pd.read_parquet(EMB_PATH)
if isinstance(emb, pd.DataFrame) and 'embedding' in emb.columns:
    emb_arr = np.stack(emb['embedding'].values).astype(np.float32)
else:
    emb_arr = emb.values.astype(np.float32)
emb_norm = emb_arr / (np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-9)

# --- Step 2: bucket size distribution ---
buckets = defaultdict(list)
for i in range(n_items):
    buckets[int(sid[i, 0])].append(i)
bucket_sizes = np.array([len(v) for v in buckets.values()])
print(f'\n[Step 2] L1 bucket distribution:')
print(f'  n_buckets total: {len(buckets)} (effective: {(bucket_sizes > 0).sum()})')
print(f'  size: min={bucket_sizes.min()}, max={bucket_sizes.max()}, mean={bucket_sizes.mean():.1f}, median={np.median(bucket_sizes):.0f}')
print(f'  size >=5: {(bucket_sizes >= 5).sum()}/{(bucket_sizes > 0).sum()} buckets, items={(bucket_sizes[bucket_sizes >= 5]).sum()}/{n_items}')
print(f'  size >=10: {(bucket_sizes >= 10).sum()} buckets, items={(bucket_sizes[bucket_sizes >= 10]).sum()}')
print(f'  size >=30: {(bucket_sizes >= 30).sum()} buckets, items={(bucket_sizes[bucket_sizes >= 30]).sum()}')
print(f'  size distribution (percentiles): 10%={np.percentile(bucket_sizes, 10):.0f}, 25%={np.percentile(bucket_sizes, 25):.0f}, 50%={np.percentile(bucket_sizes, 50):.0f}, 75%={np.percentile(bucket_sizes, 75):.0f}, 90%={np.percentile(bucket_sizes, 90):.0f}, 95%={np.percentile(bucket_sizes, 95):.0f}, 99%={np.percentile(bucket_sizes, 99):.0f}')

# --- Step 3: conditional Mantel + kNN within L1 buckets ---
MIN_BUCKET = 5
KNN = 10
all_rhos_l1 = []
all_rhos_l2 = []
all_rhos_l3 = []
all_hit_l1 = []
all_hit_l2 = []
all_hit_l3 = []
per_bucket_stats = []

print(f'\n[Step 3] Conditional Mantel + kNN (min_bucket={MIN_BUCKET})')
for bucket_id, item_ids in buckets.items():
    if len(item_ids) < MIN_BUCKET:
        continue
    item_ids = np.array(item_ids)
    n = len(item_ids)
    E_sub = emb_norm[item_ids]
    S_sub = sid[item_ids]

    # Within-bucket cos distance
    cos_d = cdist(E_sub, E_sub, metric='cosine')

    # Within-bucket SID Hamming (per layer)
    # For L1: dist = 0 if same L1 code, else 1 — but within a bucket all L1 are SAME
    # So for "conditional" Mantel we look at L2 and L3 (these can differ within bucket)
    # For Mantel L1 we'd compute "do items within same L1 have smaller cos than items in different L1?"
    # — but within a single bucket, L1 is constant. Use L2/L3 only for within-bucket Mantel.
    s_l2 = np.zeros((n, n), dtype=np.float32)
    s_l3 = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        s_l2[i] = (S_sub[:, 1] != S_sub[i, 1]).astype(np.float32)
        s_l3[i] = ((S_sub[:, 1] != S_sub[i, 1]) | (S_sub[:, 2] != S_sub[i, 2])).astype(np.float32)

    iu = np.triu_indices(n, k=1)
    cos_flat = cos_d[iu]
    s_l2_flat = s_l2[iu]
    s_l3_flat = s_l3[iu]

    if len(cos_flat) > 100:
        rho_l2 = spearmanr(cos_flat, s_l2_flat).correlation
        rho_l3 = spearmanr(cos_flat, s_l3_flat).correlation
    else:
        rho_l2, rho_l3 = np.nan, np.nan

    # kNN within bucket
    nbrs = NearestNeighbors(n_neighbors=min(KNN+1, n), metric='cosine', algorithm='brute').fit(E_sub)
    _, idx = nbrs.kneighbors(E_sub)
    # Exclude self
    idx = idx[:, 1:]
    if idx.shape[1] == 0:
        continue
    nbrs_sid = S_sub[idx]
    target_sid = S_sub[:, None, :]  # (n, 1, n_layers)

    h_l1 = (nbrs_sid[:, :, 0] == target_sid[:, :, 0]).sum(axis=1).mean() / min(KNN, n-1)
    h_l2 = ((nbrs_sid[:, :, 0] == target_sid[:, :, 0]) & (nbrs_sid[:, :, 1] == target_sid[:, :, 1])).sum(axis=1).mean() / min(KNN, n-1)
    h_l3 = ((nbrs_sid[:, :, 0] == target_sid[:, :, 0]) & (nbrs_sid[:, :, 1] == target_sid[:, :, 1]) & (nbrs_sid[:, :, 2] == target_sid[:, :, 2])).sum(axis=1).mean() / min(KNN, n-1)

    all_rhos_l2.append(rho_l2)
    all_rhos_l3.append(rho_l3)
    all_hit_l1.append(h_l1)
    all_hit_l2.append(h_l2)
    all_hit_l3.append(h_l3)
    per_bucket_stats.append({
        'bucket_id': int(bucket_id),
        'size': int(n),
        'rho_l2': float(rho_l2) if not np.isnan(rho_l2) else None,
        'rho_l3': float(rho_l3) if not np.isnan(rho_l3) else None,
        'hit_l1': float(h_l1),
        'hit_l2': float(h_l2),
        'hit_l3': float(h_l3),
    })

print(f'  Buckets analyzed: {len(all_rhos_l2)}')
mean_rho_l2 = float(np.nanmean(all_rhos_l2))
mean_rho_l3 = float(np.nanmean(all_rhos_l3))
mean_hit_l1 = float(np.mean(all_hit_l1))
mean_hit_l2 = float(np.mean(all_hit_l2))
mean_hit_l3 = float(np.mean(all_hit_l3))
print(f'  Conditional (within L1 bucket) mean Mantel rho:')
print(f'    L2: {mean_rho_l2:.4f}')
print(f'    L3: {mean_rho_l3:.4f}')
print(f'  Conditional (within L1 bucket) mean kNN hit rate (k=10):')
print(f'    L1 (constant within bucket, should be 1.0): {mean_hit_l1:.4f}')
print(f'    L2: {mean_hit_l2:.4f}')
print(f'    L3: {mean_hit_l3:.4f}')

# --- Comparison: global vs conditional ---
baseline = json.load(open(BASELINE_PATH))
g_l2 = baseline['mantel_global']['rho_l2']
g_l3 = baseline['mantel_global']['rho_l3']
g_hit_l2 = baseline['knn_global_k10']['hit_l2']
g_hit_l3 = baseline['knn_global_k10']['hit_l3']

print(f'\n=== Global vs Conditional ===')
print(f'Mantel L2:  global={g_l2:.4f}  vs  conditional={mean_rho_l2:.4f}  (delta={mean_rho_l2-g_l2:+.4f})')
print(f'Mantel L3:  global={g_l3:.4f}  vs  conditional={mean_rho_l3:.4f}  (delta={mean_rho_l3-g_l3:+.4f})')
print(f'kNN L2:     global={g_hit_l2:.4f}  vs  conditional={mean_hit_l2:.4f}  (delta={mean_hit_l2-g_hit_l2:+.4f})')
print(f'kNN L3:     global={g_hit_l3:.4f}  vs  conditional={mean_hit_l3:.4f}  (delta={mean_hit_l3-g_hit_l3:+.4f})')

out = {
    'task': 'task69_step23_conditional',
    'min_bucket': MIN_BUCKET,
    'n_buckets_analyzed': len(all_rhos_l2),
    'bucket_size_distribution': {
        'min': int(bucket_sizes.min()),
        'max': int(bucket_sizes.max()),
        'mean': float(bucket_sizes.mean()),
        'median': float(np.median(bucket_sizes)),
        'p10': float(np.percentile(bucket_sizes, 10)),
        'p25': float(np.percentile(bucket_sizes, 25)),
        'p50': float(np.percentile(bucket_sizes, 50)),
        'p75': float(np.percentile(bucket_sizes, 75)),
        'p90': float(np.percentile(bucket_sizes, 90)),
        'p95': float(np.percentile(bucket_sizes, 95)),
        'p99': float(np.percentile(bucket_sizes, 99)),
    },
    'conditional_results': {
        'mean_rho_l2': mean_rho_l2,
        'mean_rho_l3': mean_rho_l3,
        'mean_hit_l1_within_bucket': mean_hit_l1,
        'mean_hit_l2_within_bucket': mean_hit_l2,
        'mean_hit_l3_within_bucket': mean_hit_l3,
    },
    'comparison_global_vs_conditional': {
        'mantel_l2_delta': float(mean_rho_l2 - g_l2),
        'mantel_l3_delta': float(mean_rho_l3 - g_l3),
        'knn_l2_delta': float(mean_hit_l2 - g_hit_l2),
        'knn_l3_delta': float(mean_hit_l3 - g_hit_l3),
    },
    'per_bucket_stats_sample': per_bucket_stats[:20],
}
out_path = os.path.join(OUT_DIR, 'result.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
print(f'\nSaved → {out_path}')
