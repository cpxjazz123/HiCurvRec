#!/usr/bin/env python3
"""Idea 1 WF 综合评估 — A_baseline vs post-hoc WF vs retrained WF

加载三组 rqidx:
- A_baseline (K=[256,256,256])
- post-hoc WF (K=[256,64,16] from merge, no retraining)
- retrained WF (K=[256,64,16] from idea1_WF_inference.py)

对比指标:
1. 碰撞率 (unique SIDs / N)
2. 每层 residual norm 比值
3. V-info per layer (I(r_l → Y))
4. Brand MI per layer (Idea 3 联动)
5. CF pair distance per layer

判定:
- retrained WF 是否 > post-hoc WF (重训价值)
- retrained WF 是否 ≈ A_baseline (WF 没有坏处但也没有好处)
- retrained WF 是否 < A_baseline (WF 失败, 但现象 1 仍 PASS)
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
CF_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/cf_ppmi_svd256.pt'

IDX_FILES = {
    'A_baseline':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'post-hoc WF': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_posthoc_rqidx.pt',
    'retrained WF': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def empirical_mi_bits(joint):
    N = joint.sum()
    pxy = joint / N
    px = joint.sum(axis=1) / N
    py = joint.sum(axis=0) / N
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
    return float(mi / np.log(2))


def fast_mi_bits(x, y, n_sample=2000, seed=42, n_neighbors=5):
    if x.ndim == 2 and x.shape[1] > 32:
        x = PCA(n_components=16, random_state=seed).fit_transform(x)
    N = x.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    mi_nats = mutual_info_classif(x_s, y_s, n_neighbors=n_neighbors,
                                   random_state=seed, discrete_features=False, n_jobs=4)
    return float(mi_nats.mean() / np.log(2))


def main():
    print('=' * 70)
    print('Idea 1 WF 综合评估 — A_baseline vs post-hoc WF vs retrained WF')
    print('=' * 70)

    # Build Y from K-Means on raw embedding
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    N = x.shape[0]
    km_y = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3,
                           batch_size=1024).fit(x)
    y = km_y.labels_.astype(np.int64)

    # Build brand labels (Top-150)
    metadata = json.load(open(META_PATH))
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in range(N)])
    cnt = Counter(brand_arr.tolist())
    top_brands = [b for b, _ in cnt.most_common(150) if b != 'NO_BRAND']
    brand_labels = np.array([b if b in top_brands else 'OTHER' for b in brand_arr])
    brand_subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(brand_subs)}
    brand_ids = np.array([brand_to_id[b] for b in brand_labels])
    H_brand = float(-np.sum((p := np.bincount(brand_ids)/N) * np.log2(p + 1e-12)))
    print(f'  H(brand_150) = {H_brand:.4f} bits')

    # Load CF
    cf = torch.load(CF_PATH, map_location='cpu', weights_only=False)
    f_norm = cf['f_normalized'].numpy().astype(np.float32)
    print(f'  CF shape: {f_norm.shape}')

    # Per-algorithm metrics
    all_results = {}
    for name, path in IDX_FILES.items():
        print(f'\n--- {name} ---')
        if not os.path.exists(path):
            print(f'  FILE NOT FOUND: {path}, skip')
            continue
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        idx_lst = bundle['idx_lst']
        sids = bundle.get('sids')
        n_unique = int(np.unique(sids.numpy() if torch.is_tensor(sids) else sids, axis=0).shape[0]) if sids is not None else None
        collision = 1 - n_unique / N if n_unique else None
        print(f'  N items: {N}, unique SIDs: {n_unique}, collision: {collision:.4f}' if collision else f'  no sids')

        # K_per_layer
        Ks = [int(idx_lst[l].max()) + 1 for l in range(len(idx_lst))] if idx_lst else None
        print(f'  K per layer: {Ks}')

        algo_res = {'K_per_layer': Ks, 'n_unique_SIDs': n_unique, 'collision_rate': collision}

        # V-info per layer
        mi_v = []
        for l in range(1, len(r_lst)):  # r[0] is x itself
            r_l = r_lst[l].numpy().astype(np.float32)
            mi = fast_mi_bits(r_l, y)
            mi_v.append(mi)
            print(f'  V-info I(r_{l} → Y) = {mi:.4f} bits')
        algo_res['V_info_per_layer'] = mi_v

        # Brand MI per layer (using q_l for K=256)
        brand_mi = []
        for l in range(len(idx_lst)):
            q_l = q_lst[l].numpy().astype(np.float32)
            K_l = min(int(idx_lst[l].max()) + 1, 256)
            km = MiniBatchKMeans(n_clusters=K_l, random_state=42, n_init=3,
                                 batch_size=1024).fit(q_l)
            c_q = km.labels_.astype(np.int64)
            joint = np.zeros((K_l, len(brand_subs)), dtype=np.int64)
            np.add.at(joint, (c_q, brand_ids), 1)
            mi = empirical_mi_bits(joint)
            brand_mi.append(mi)
            print(f'  Brand MI I(q_{l+1} → brand) = {mi:.4f} bits')
        algo_res['Brand_MI_per_layer'] = brand_mi

        # Residual norm ratio per layer
        r_norms = [float(r_lst[l].norm(dim=-1).mean()) for l in range(len(r_lst))]
        algo_res['r_norm_per_layer'] = r_norms
        algo_res['r_norm_ratio'] = [r_norms[l+1] / (r_norms[0] + 1e-12) for l in range(len(r_lst) - 1)]

        # CF pair distance per layer
        cf_dists = []
        rng = np.random.default_rng(42)
        n_pairs = 30000
        i_idx = rng.integers(0, N, size=n_pairs)
        j_idx = rng.integers(0, N, size=n_pairs)
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        sim_pairs = (f_norm[i_idx] * f_norm[j_idx]).sum(axis=1)
        tau_high = float(np.quantile(sim_pairs, 0.90))
        tau_low = float(np.quantile(sim_pairs, 0.10))
        high_mask = sim_pairs > tau_high
        low_mask = sim_pairs < tau_low
        for l in range(len(q_lst)):
            q_l = q_lst[l].numpy().astype(np.float32)
            d_high = np.linalg.norm(q_l[i_idx[high_mask]] - q_l[j_idx[high_mask]], axis=1)
            d_low  = np.linalg.norm(q_l[i_idx[low_mask]]  - q_l[j_idx[low_mask]],  axis=1)
            delta = float(d_low.mean() - d_high.mean())
            cf_dists.append(delta)
            print(f'  CF pair Δ (L{l+1}) = {delta:+.4f}')
        algo_res['CF_pair_delta_per_layer'] = cf_dists

        all_results[name] = algo_res

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea 1 WF 完整对比')
    print('=' * 70)
    if 'A_baseline' in all_results and 'retrained WF' in all_results:
        a = all_results['A_baseline']
        w = all_results['retrained WF']
        # V-info comparison
        for l in range(len(a['V_info_per_layer'])):
            av, wv = a['V_info_per_layer'][l], w['V_info_per_layer'][l]
            print(f'  V-info L{l+1}: AQ={av:.4f}, WF={wv:.4f}, '
                  f'Δ={(wv-av):+.4f}')
        # Brand MI
        for l in range(len(a['Brand_MI_per_layer'])):
            ab, wb = a['Brand_MI_per_layer'][l], w['Brand_MI_per_layer'][l]
            print(f'  Brand MI L{l+1}: AQ={ab:.4f}, WF={wb:.4f}, '
                  f'Δ={(wb-ab):+.4f}')
        # Collision rate
        ac = a['collision_rate']
        wc = w['collision_rate']
        if ac is not None and wc is not None:
            print(f'  Collision: AQ={ac:.4f}, WF={wc:.4f}')
        else:
            print(f'  Collision: AQ=None (没有 sids 数据), WF={wc:.4f}')

    # Save
    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    out_path = os.path.join(OUT_DIR, 'idea1_WF_full_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'H_brand': H_brand,
            'results': all_results,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()