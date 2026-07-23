#!/usr/bin/env python3
"""Task #69 Step 1: Global baseline (Version A) on Task #126 phonism RQ-VAE.

复现 task380/390/27 历史 Mantel + kNN hit rate 实验, 但在 phonism RQ-VAE
(sentence-t5-base 768d, 11924 items, 3 layers × 256 codes) 上跑.

为 Step 2-3 条件化对比提供 baseline.

Output:
- /home/wlia0047/ar57/wenyu/GeneRec/result/task69_step1_global/baseline.json
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/products/task126_phonism_rqvae/semantic_ids.pt'
EMB_PATH = '/home/wlia0047/ar57/wenyu/genrec/dataset/amazon/processed/toys/item_emb_sentence-t5-base.parquet'
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/result/task69_step1_global'
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('Task #69 Step 1: Global baseline Mantel + kNN on Task #126 phonism')
print('=' * 70)

# Load SID
sid_dict = torch.load(SID_PATH, weights_only=False)
sid = sid_dict['semantic_ids'].numpy()  # (11924, 3)
n_items, n_layers = sid.shape
K = 256
print(f'SID: {n_items} items x {n_layers} layers, codebook={K}')
print(f'Unique per layer: {[len(np.unique(sid[:, i])) for i in range(n_layers)]}')

# Load embedding
emb = pd.read_parquet(EMB_PATH)
if isinstance(emb, pd.DataFrame):
    if 'embedding' in emb.columns:
        emb_arr = np.stack(emb['embedding'].values)
    else:
        emb_arr = emb.values
else:
    emb_arr = emb
print(f'Embedding shape: {emb_arr.shape}, dtype: {emb_arr.dtype}')
emb_arr = emb_arr.astype(np.float32)

# Normalize embedding for cosine distance
emb_norm = emb_arr / (np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-9)

# --- SID tree distance ---
def sid_tree_dist(sid_codes, layer):
    """For each pair, count number of shared prefix length up to `layer` layers.
    Convert to distance: 1 if no shared prefix, 0 if all match.
    Distance = number of layers where codes differ (Hamming on first `layer` digits).
    """
    n = sid_codes.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for l in range(layer):
        col = sid_codes[:, l:l+1]
        diff = (col != col.T).astype(np.float32)
        D += diff
    return D

print('\n[computing SID tree distances]')
sid_d_l1 = sid_tree_dist(sid, 1)  # (11924, 11924) — L1 digit match
sid_d_l2 = sid_tree_dist(sid, 2)  # L1+L2
sid_d_l3 = sid_tree_dist(sid, 3)  # L1+L2+L3

print('SID dist range:')
print(f'  L1: 0={int((sid_d_l1==0).sum())} pairs, 1={int((sid_d_l1==1).sum())} pairs')
print(f'  L2: max={sid_d_l2.max():.0f}, mean={sid_d_l2.mean():.3f}')
print(f'  L3: max={sid_d_l3.max():.0f}, mean={sid_d_l3.mean():.3f}')

# --- Global Mantel test ---
# Sample 2000 items for tractability (mirroring task380)
print('\n[global Mantel test, n=2000 sample]')
rng = np.random.default_rng(42)
sample_idx = rng.choice(n_items, 2000, replace=False)
E_sub = emb_norm[sample_idx]
emb_dist_sub = cdist(E_sub, E_sub, metric='cosine')
sid_d_l1_sub = sid_d_l1[np.ix_(sample_idx, sample_idx)]
sid_d_l2_sub = sid_d_l2[np.ix_(sample_idx, sample_idx)]
sid_d_l3_sub = sid_d_l3[np.ix_(sample_idx, sample_idx)]

# Use upper triangle only
iu = np.triu_indices(2000, k=1)
e_flat = emb_dist_sub[iu]
s_l1_flat = sid_d_l1_sub[iu]
s_l2_flat = sid_d_l2_sub[iu]
s_l3_flat = sid_d_l3_sub[iu]

mantel_l1 = spearmanr(e_flat, s_l1_flat)
mantel_l2 = spearmanr(e_flat, s_l2_flat)
mantel_l3 = spearmanr(e_flat, s_l3_flat)
print(f'  Mantel rho (n=2000):')
print(f'    L1 vs cos: rho={mantel_l1.correlation:.4f}, p={mantel_l1.pvalue:.2e}')
print(f'    L2 vs cos: rho={mantel_l2.correlation:.4f}, p={mantel_l2.pvalue:.2e}')
print(f'    L3 vs cos: rho={mantel_l3.correlation:.4f}, p={mantel_l3.pvalue:.2e}')

# --- Global kNN hit rate (k=10) ---
# For each item, find top-10 nearest by global cosine, then check how many share L1/L2/L3
print('\n[global kNN hit rate, k=10]')
KNN = 10
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=KNN+1, metric='cosine', algorithm='brute').fit(emb_norm)
distances, indices = nbrs.kneighbors(emb_norm)
# Exclude self (index 0)
neighbors = indices[:, 1:]  # (11924, KNN)

hit_l1 = []
hit_l2 = []
hit_l3 = []
for i in range(n_items):
    nbr_l1_match = (sid[neighbors[i], 0] == sid[i, 0]).sum()
    nbr_l2_match = ((sid[neighbors[i], 0] == sid[i, 0]) & (sid[neighbors[i], 1] == sid[i, 1])).sum()
    nbr_l3_match = ((sid[neighbors[i], 0] == sid[i, 0]) & (sid[neighbors[i], 1] == sid[i, 1]) & (sid[neighbors[i], 2] == sid[i, 2])).sum()
    hit_l1.append(nbr_l1_match / KNN)
    hit_l2.append(nbr_l2_match / KNN)
    hit_l3.append(nbr_l3_match / KNN)

hit_l1_mean = float(np.mean(hit_l1))
hit_l2_mean = float(np.mean(hit_l2))
hit_l3_mean = float(np.mean(hit_l3))
print(f'  Global kNN (k=10) hit rate:')
print(f'    L1 share: {hit_l1_mean:.4f}')
print(f'    L2 share: {hit_l2_mean:.4f}')
print(f'    L3 share: {hit_l3_mean:.4f}')

# Save baseline
out = {
    'task': 'task69_step1_global',
    'data': {
        'sid_path': SID_PATH,
        'emb_path': EMB_PATH,
        'n_items': int(n_items),
        'n_layers': int(n_layers),
        'codebook_size': int(K),
        'coverage_per_layer': [len(np.unique(sid[:, i])) for i in range(n_layers)],
        'unique_3token_sids': int(len(np.unique(sid, axis=0))),
    },
    'mantel_global': {
        'n_sample': 2000,
        'rho_l1': float(mantel_l1.correlation),
        'rho_l2': float(mantel_l2.correlation),
        'rho_l3': float(mantel_l3.correlation),
        'pvalue_l1': float(mantel_l1.pvalue),
        'pvalue_l2': float(mantel_l2.pvalue),
        'pvalue_l3': float(mantel_l3.pvalue),
    },
    'knn_global_k10': {
        'hit_l1': hit_l1_mean,
        'hit_l2': hit_l2_mean,
        'hit_l3': hit_l3_mean,
    },
}
out_path = os.path.join(OUT_DIR, 'baseline.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved baseline → {out_path}')
