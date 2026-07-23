#!/usr/bin/env python3
"""Idea B 现象 2: causal 正交下重做 Idea 2 现象 2 的去冗余操作

欧氏版: r_l^⊥ = r_l - q_1 · [(q_1.T r_l) / (q_1.T q_1)]
Causal版: r_l^{⊥,M} = r_l - q_1 · [(q_1.T M r_l) / (q_1.T M q_1)]

测 ΔI_V^{decouple-causal}(l) vs ΔI_V^{decouple-Eucl}(l)

M 构造: RQ 码本第一层 centroids 协方差 (placeholder; 现象 3 详细对比)
"""

import sys
sys.path.insert(0, '/home/wlia0047/arenyu/GeneRec/GRID' if False else '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# Pre-import to break circular import
import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}
RQ_CKPT_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-11/13-15-28/checkpoints/checkpoint_000_003000.ckpt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
}

N_MI_SAMPLE = 2000
PCA_RED_DIM = 16


def fast_mi_bits(x, y, n_sample=N_MI_SAMPLE, seed=42, n_neighbors=5):
    if x.ndim == 2 and x.shape[1] > 32:
        x = PCA(n_components=PCA_RED_DIM, random_state=seed).fit_transform(x)
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


def build_y(emb_path, K=20):
    x = torch.load(emb_path, map_location='cpu', weights_only=False).float()
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024).fit(x.numpy())
    return km.labels_.astype(np.int64)


def load_M_from_ckpt(ckpt_path, layer=0):
    """M = centroids.T @ centroids (D x D covariance proxy)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    key = f'quantization_layer_list.{layer}.centroids'
    if key not in sd:
        for k in sd:
            if 'centroid' in k.lower() and str(layer) in k:
                key = k
                break
    t = sd[key].float().numpy()  # (D, K) or (K, D)
    print(f'  raw {key}: shape={t.shape}')
    # Ensure t is (D, K): if first dim is K (256), transpose
    if t.shape[0] == 256:
        t = t.T  # now (D, K) = (2048, 256)
    print(f'  using shape={t.shape}')
    return t @ t.T  # (D, D) = (2048, 2048)


def decouple_eucl(r_l, q_1):
    """r_l^⊥ = r_l - q_1 @ [(q_1.T q_1)^-1 @ q_1.T r_l]"""
    beta = np.linalg.pinv(q_1.T @ q_1 + 1e-4 * np.eye(q_1.shape[1])) @ q_1.T @ r_l
    return r_l - q_1 @ beta


def decouple_causal(r_l, q_1, M):
    """r_l^{⊥,M} = r_l - q_1 · [(q_1.T M r_l) / (q_1.T M q_1)] (per item)
    Vectorized: project each row onto q_1 in M-metric.
    For each item i:
        proj_coef = (q_1[i].T M r_l[i]) / (q_1[i].T M q_1[i])
        r_perp[i] = r_l[i] - proj_coef * q_1[i]
    """
    # Numerator per item: (M r_l[i]).T q_1[i] = r_l[i].T M q_1[i]
    #   -> vector: diag(q_1 @ M @ r_l.T) — expensive if N large
    # We can compute: Mr_l = r_l @ M.T (since M symmetric)
    Mr_l = r_l @ M.T  # (N, D)
    Mq_1 = M @ q_1.T  # (D, N)
    # numerator[i] = q_1[i] . M . r_l[i] = (Mr_l[i] * q_1[i]).sum() = (Mr_l * q_1).sum(axis=1)
    num = (Mr_l * q_1).sum(axis=1)  # (N,)
    # denominator[i] = q_1[i] . M . q_1[i] = (Mq_1[:, i] * q_1[i]).sum()
    # Mq_1 is (D, N): column i = M q_1[i]
    den = (Mq_1 * q_1.T).sum(axis=0)  # (N,)
    proj_coef = num / (den + 1e-12)
    return r_l - proj_coef[:, None] * q_1


def main():
    print('=' * 70)
    print('Loading Y proxy and M for each algo')
    print('=' * 70)
    y = build_y(EMB_PATH, K=20)
    M_dict = {}
    for name, ckpt in RQ_CKPT_PATHS.items():
        if os.path.exists(ckpt):
            M_dict[name] = load_M_from_ckpt(ckpt, layer=0)
            print(f'  {name} M: trace={np.trace(M_dict[name]):.2e}, '
                  f'frobenius={np.linalg.norm(M_dict[name]):.2e}')
        else:
            print(f'  {name} ckpt missing: {ckpt}')

    print('\n' + '=' * 70)
    print('Loading r_lst, q_1 for A/B/C')
    print('=' * 70)
    bundles = {n: torch.load(p, map_location='cpu', weights_only=False)
               for n, p in RQIDX_PATHS.items()}

    results = {}
    for name in bundles:
        if name not in M_dict:
            continue
        print(f'\n--- {name} ---')
        bundle = bundles[name]
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        M = M_dict[name]
        algo_res = {}
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            q_1 = q_lst[0].numpy().astype(np.float32)
            r_perp_e = decouple_eucl(r_l, q_1)
            r_perp_m = decouple_causal(r_l, q_1, M)
            mi_base   = fast_mi_bits(r_l, y)
            mi_perp_e = fast_mi_bits(r_perp_e, y)
            mi_perp_m = fast_mi_bits(r_perp_m, y)
            delta_e = mi_perp_e - mi_base
            delta_m = mi_perp_m - mi_base
            # Causal gain over euclidean: difference
            causal_gain = delta_m - delta_e
            algo_res[f'l{l}'] = {
                'v_info_base':         mi_base,
                'v_info_perp_eucl':    mi_perp_e,
                'v_info_perp_causal':  mi_perp_m,
                'delta_eucl':          delta_e,
                'delta_causal':        delta_m,
                'causal_gain':         causal_gain,
                'r_norm':              float(np.linalg.norm(r_l, axis=-1).mean()),
                'perp_eucl_norm':      float(np.linalg.norm(r_perp_e, axis=-1).mean()),
                'perp_causal_norm':    float(np.linalg.norm(r_perp_m, axis=-1).mean()),
            }
            print(f'  L{l}: I_V(r)={mi_base:.4f}, '
                  f'I_V(perp_E)={mi_perp_e:.4f} (Δ={delta_e:+.4f}), '
                  f'I_V(perp_M)={mi_perp_m:.4f} (Δ={delta_m:+.4f}), '
                  f'causal_gain={causal_gain:+.4f}')
        results[name] = algo_res

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea B 现象 2 (causal 正交是否挖出更多)')
    print('=' * 70)
    summary = {'per_algo': {}}
    for name, ar in results.items():
        all_gains = [ar[lk]['causal_gain'] for lk in ar]
        max_gain = max(all_gains)
        mean_gain = float(np.mean(all_gains))
        # Kill line: causal_gain ≈ 0 for all
        kill = '✓ causal 显著提升' if max_gain > 0.05 else \
               '✗ kill (Idea B dead)'
        print(f'  {name}: max_gain={max_gain:+.4f} mean_gain={mean_gain:+.4f}  {kill}')
        summary['per_algo'][name] = {
            'max_causal_gain': float(max_gain),
            'mean_causal_gain': mean_gain,
            'kill_verdict': kill,
        }

    out_path = os.path.join(OUT_DIR, 'ideaB_phenomenon2.json')
    with open(out_path, 'w') as f:
        json.dump({
            'M_source': 'RQ_L1_codebook_covariance',
            'results': results,
            'summary': summary,
        }, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()