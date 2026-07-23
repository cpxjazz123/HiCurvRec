#!/usr/bin/env python3
"""Idea 1 Post-hoc WF: 不用重训, 直接 merge A_baseline codebooks [256,256,256] → [256,64,16]

核心思想:
- A_baseline 训的是 K=256 各层. WF 启发: 把 L1 256 保留, L2 256 merge 到 64, L3 256 merge 到 16.
- K-Means merge 每层 centroids 自身 (对 centroid 做聚类)
- 重 assign r_l → q_l (基于 merged codebook)

优势: 0 成本, 立刻看到 WF 分配有什么效果
劣势: 不是 retrained optimal, 只是现有 centroids 的 sub-sampling 近似
但仍能 give a 'WF-shaped' 端到端估计.

输出: idea1_WF_posthoc_rqidx.pt (与 infer_diag.json)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import KMeans

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
A_CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'
# WF target
WF_K = [256, 64, 16]


def merge_codebook(centroids, K_target):
    """Cluster 256 centroids into K_target super-centroids via K-Means."""
    if centroids.shape[0] == K_target:
        return centroids.copy()
    km = KMeans(n_clusters=K_target, random_state=42, n_init=10).fit(centroids)
    return km.cluster_centers_.astype(np.float32)


def main():
    print('=' * 70)
    print('Idea 1 Post-hoc WF: codebook merge from A_baseline')
    print('=' * 70)
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    print(f'  x shape: {x.shape}')

    ck = torch.load(A_CKPT, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    orig_codebooks = []
    for l in range(3):
        c = sd[f'quantization_layer_list.{l}.centroids'].float().numpy()
        print(f'  L{l+1} orig codebook shape: {c.shape}, mean_norm={np.linalg.norm(c, axis=1).mean():.3f}')
        orig_codebooks.append(c)

    # Merge centroids to [256, 64, 16]
    new_codebooks = []
    for l in range(3):
        new_c = merge_codebook(orig_codebooks[l], WF_K[l])
        new_codebooks.append(new_c)
        print(f'  L{l+1} merged codebook shape: {new_c.shape}, mean_norm={np.linalg.norm(new_c, axis=1).mean():.3f}')

    # Re-do RKMeans forward with merged codebooks
    r_lst = [x.copy()]
    q_lst, idx_lst = [], []
    for l in range(3):
        r = r_lst[-1]
        C = new_codebooks[l]
        # Squared Euclidean
        r_norm2 = (r ** 2).sum(axis=-1, keepdims=True)
        c_norm2 = (C ** 2).sum(axis=-1)
        cross = r @ C.T
        d2 = r_norm2 + c_norm2[None, :] - 2 * cross
        idx = d2.argmin(axis=-1)
        q = C[idx]
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
        print(f'  L{l+1}: idx_range=[0,{idx.max()}], active_codes={len(np.unique(idx))}, '
              f'r_after_norm={(r-q).std():.4f}')

    sids = np.stack([idx_lst[0], idx_lst[1], idx_lst[2]], axis=-1)
    n_unique = len(np.unique(sids, axis=0))
    n_total = sids.shape[0]
    collision_rate = 1 - n_unique / n_total
    print(f'\n  SIDs: {n_unique}/{n_total} = {n_unique/n_total:.4f}, '
          f'collision rate: {collision_rate:.4f}')

    # Save as torch.load-able
    out = {
        'r_lst': [torch.tensor(r) for r in r_lst],
        'q_lst': [torch.tensor(q) for q in q_lst],
        'idx_lst': [torch.tensor(idx) for idx in idx_lst],
        'codebooks': [torch.tensor(c) for c in new_codebooks],
        'gains': [None, None, None],
        'sids': torch.tensor(sids),
        'K_per_layer': WF_K,
        'collision_rate': float(collision_rate),
        'n_unique': int(n_unique),
        'n_total': int(n_total),
    }
    out_path = os.path.join(OUT_DIR, 'idea1_WF_posthoc_rqidx.pt')
    torch.save(out, out_path)
    print(f'Saved → {out_path}')

    diag = {
        'algo': 'idea1_WF_posthoc_K_256_64_16',
        'K_per_layer': WF_K,
        'N_items': int(x.shape[0]),
        'n_unique_SIDs': int(n_unique),
        'collision_rate': float(collision_rate),
        'r_norm_per_layer': [float(np.linalg.norm(r_lst[l+1], axis=-1).mean()) for l in range(3)],
        'q_norm_per_layer': [float(np.linalg.norm(q_lst[l], axis=-1).mean()) for l in range(3)],
        'method': 'codebook merge via K-Means (post-hoc)',
    }
    diag_path = os.path.join(OUT_DIR, 'idea1_WF_posthoc_diag.json')
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f'Saved diag → {diag_path}')


if __name__ == '__main__':
    main()