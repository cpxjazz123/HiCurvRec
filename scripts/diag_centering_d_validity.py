#!/usr/bin/env python3
"""Centering-D validity check + U_brand vs PCA overlap

Part 1: cos(mean(q_1), mean(x)) for all 4+2 algorithms + ρ(x_c) vs ρ(q_1_c) audit
Part 2: U_brand top-10 vs PCA top-k (k=10, 50, 100, 256) overlap; identify
        whether U_brand is mostly the embedding's top principal directions
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import time
import numpy as np
import torch
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/centering_validity'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'

RQIDX = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'idea1_WF':   '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
    'HRQ':        '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/hrq_rqidx.pt',
    'AQ':         '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/aq_rqidx.pt',
}


def build_task_subspace_fast(emb_c_np, labels_str, m=10, min_count=2):
    d = emb_c_np.shape[1]
    cnt = Counter(labels_str)
    valid_labels = sorted([k for k, v in cnt.items() if v >= min_count])
    N = emb_c_np.shape[0]
    mu_G = emb_c_np.mean(axis=0)
    diffs = np.zeros((len(valid_labels), d))
    for i, k in enumerate(valid_labels):
        mask = np.array([l == k for l in labels_str])
        n_k = mask.sum()
        if n_k == 0:
            continue
        diffs[i] = np.sqrt(n_k / N) * (emb_c_np[mask].mean(axis=0) - mu_G)
    S_small = diffs @ diffs.T
    eigvals, V_small = np.linalg.eigh(S_small)
    idx = np.argsort(-np.abs(eigvals))[:m]
    U = diffs.T @ V_small[:, idx]
    Q, _ = np.linalg.qr(U)
    return Q[:, :m]


def subspace_overlap(U1, U2):
    m = U1.shape[1]
    cos_mat = np.abs(U1.T @ U2)
    return float(cos_mat.max(axis=1).mean())


def rho(v, U):
    v_proj = v @ U
    var_task = (v_proj ** 2).sum(axis=1).mean()
    var_total = (v ** 2).sum(axis=1).mean()
    return float(var_task / var_total) if var_total > 0 else 0.0


def main():
    t0 = time.time()
    print('=' * 70)
    print('Centering-D validity + U_brand vs PCA overlap')
    print('=' * 70)

    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float().numpy()
    N, d = emb.shape
    with open(META_PATH) as f:
        md = json.load(f)
    labels_brand = [md[str(i)]['brand'] for i in range(N)]

    emb_centered = emb - emb.mean(0)
    mu_x = emb.mean(0)
    mu_x_norm = np.linalg.norm(mu_x)
    print(f'\n[Part 1] Centering-D validity — cos(mean(q_1), mean(x)) per algorithm')
    print(f'  ‖mean(x)‖ = {mu_x_norm:.4f}')
    print(f'  {"algo":<14} {"‖mean(q1)‖":<14} {"cos(m_q1,m_x)":<14} {"ratio m_q1/m_x":<14} {"ρ(x_c)":<10} {"ρ(q1_c)":<10} {"ρ_drop":<10}')

    rows = []
    for algo, path in RQIDX.items():
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        q1 = bundle['q_lst'][0].float().numpy()
        mu_q = q1.mean(0)
        cos = float(mu_q @ mu_x / (np.linalg.norm(mu_q) * mu_x_norm))
        ratio_norm = float(np.linalg.norm(mu_q) / mu_x_norm)
        q1_centered = q1 - mu_q  # centering D
        U_brand = build_task_subspace_fast(emb_centered, labels_brand, m=10, min_count=2)
        r_x = rho(emb_centered, U_brand)
        r_q = rho(q1_centered, U_brand)
        drop = (r_x - r_q) / r_x if r_x > 0 else 0
        rows.append({
            'algo': algo,
            'mean_q1_norm': float(np.linalg.norm(mu_q)),
            'cos_mean_q1_mean_x': cos,
            'ratio_norm': ratio_norm,
            'rho_x_c_brand': r_x,
            'rho_q1_c_brand': r_q,
            'rho_drop': drop,
        })
        print(f'  {algo:<14} {np.linalg.norm(mu_q):<14.4f} {cos:<14.4f} {ratio_norm:<14.4f} '
              f'{r_x:<10.4f} {r_q:<10.4f} {drop*100:<10.1f}%')

    print('\nCentering-D validity verdict:')
    cos_values = [r['cos_mean_q1_mean_x'] for r in rows]
    print(f'  cos mean across algos: {np.mean(cos_values):.4f} ± {np.std(cos_values):.4f}')
    if np.mean(np.abs(cos_values)) > 0.95:
        print('  ✓ Centering-D justified: mean(q_1) ≈ mean(x), symmetric centering is valid')
    elif np.mean(np.abs(cos_values)) > 0.7:
        print('  ⚠ Centering-D partially valid: mean directions close but not identical')
    else:
        print('  ✗ Centering-D NOT justified: mean directions diverge, ρ comparison still biased')

    # ===== Part 2: U_brand vs PCA top-k overlap =====
    print(f'\n[Part 2] U_brand top-10 vs PCA top-k overlap')
    # PCA on centered x (x_c), so PCA captures variance direction not mean
    print('  Computing PCA on x_centered...', flush=True)
    Xc = emb_centered
    # Use randomized PCA via torch SVD for speed
    # Or sklearn PCA
    from sklearn.decomposition import PCA
    pca_full = PCA(n_components=256, random_state=42).fit(Xc)
    print(f'  PCA done in {time.time()-t0:.0f}s, top-10 explained_var: {pca_full.explained_variance_ratio_[:10].sum():.3f}')

    U_brand = build_task_subspace_fast(emb_centered, labels_brand, m=10, min_count=2)

    print(f'\n  {"k":<6} {"overlap_brand_top10_vs_PCA_topk":<35} {"PCA_explained":<15}')
    pca_results = []
    for k in [10, 20, 50, 100, 256]:
        V_k = pca_full.components_[:k].T  # (d, k)
        ov = subspace_overlap(U_brand, V_k)
        expl = pca_full.explained_variance_ratio_[:k].sum()
        pca_results.append({'k': k, 'overlap': ov, 'pca_explained': float(expl)})
        print(f'  {k:<6} {ov:<35.4f} {expl:<15.4f}')

    # Also: overlap of U_brand with itself (sanity)
    print(f'\n  Sanity: U_brand vs itself = {subspace_overlap(U_brand, U_brand):.4f} (should be 1.0)')

    # Save
    out = {
        'part1_centering_validity': rows,
        'part2_pca_overlap': pca_results,
    }
    out_path = os.path.join(OUT_DIR, 'centering_d_and_pca_overlap.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')
    print(f'Elapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()