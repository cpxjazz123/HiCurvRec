#!/usr/bin/env python3
"""Idea 1 现象 1: 深层残差空间协同结构检验

检验深层残差 r_l^(i) - r_l^(j) 距离是否与 PPMI 协同相似度有关:
- 高协同组 P_high = {(i,j) : sim_cf(i,j) > τ_high}
- 低协同组 P_low  = {(i,j) : sim_cf(i,j) < τ_low}
- Δ_l = D̄_l(P_low) - D̄_l(P_high) > 0?

如果 Δ_l 显著为正 → 深层残差空间保留协同结构 → 进入 Idea 1 现象 2
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

CF_PATH = os.path.join(OUT_DIR, 'cf_ppmi_svd256.pt')
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}

# τ thresholds: percentile-based to balance group sizes
TAU_HIGH_PCTL = 90     # top 10% sim_cf pairs
TAU_LOW_PCTL = 10      # bottom 10% sim_cf pairs
N_PAIR_SAMPLE = 30000  # subsample pairs for distance compute


def main():
    print('=' * 70)
    print('Loading PPMI-SVD collaborative embedding')
    print('=' * 70)
    cf_data = torch.load(CF_PATH, map_location='cpu', weights_only=False)
    f = cf_data['f_normalized'].numpy()  # (11924, 256)
    N = f.shape[0]
    print(f'  f shape={f.shape}, N={N}')

    print('\n' + '=' * 70)
    print(f'Step 1: Sample {N_PAIR_SAMPLE} random pairs, classify by sim_cf percentile')
    print('=' * 70)
    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, N, size=N_PAIR_SAMPLE)
    j_idx = rng.integers(0, N, size=N_PAIR_SAMPLE)
    mask_ij = i_idx != j_idx
    i_idx, j_idx = i_idx[mask_ij], j_idx[mask_ij]
    sim_cf = (f[i_idx] * f[j_idx]).sum(axis=1)
    print(f'  pair sim_cf: mean={sim_cf.mean():.4f}, std={sim_cf.std():.4f}, '
          f'p10={np.quantile(sim_cf, 0.1):.4f}, p90={np.quantile(sim_cf, 0.9):.4f}')

    tau_high = float(np.quantile(sim_cf, TAU_HIGH_PCTL / 100.0))
    tau_low  = float(np.quantile(sim_cf, TAU_LOW_PCTL / 100.0))
    high_mask = sim_cf > tau_high
    low_mask  = sim_cf < tau_low
    print(f'  tau_high (p{TAU_HIGH_PCTL}) = {tau_high:.4f}, |P_high| = {high_mask.sum()}')
    print(f'  tau_low  (p{TAU_LOW_PCTL}) = {tau_low:.4f}, |P_low| = {low_mask.sum()}')

    print('\n' + '=' * 70)
    print('Step 2: For each algorithm × each layer l=1,2,3: D̄_l(P_high) vs D̄_l(P_low)')
    print('=' * 70)
    results = {}
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        print(f'\n--- {algo_name} ---')
        bundle = torch.load(rqidx_path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']   # 4 tensors: r_0=x, r_1, r_2, r_3
        L = len(r_lst) - 1        # 3 layers

        algo_res = {'layers': {}, 'tau_high': tau_high, 'tau_low': tau_low}
        for l in range(1, L+1):
            r_l = r_lst[l].numpy()  # (11924, 2048)
            # Compute pairwise distances for high and low groups
            r_high = r_l[i_idx[high_mask]]
            r_low  = r_l[i_idx[low_mask]]
            # ||r_l^(i) - r_l^(j)||  - vectors
            diff_high = r_high - r_l[j_idx[high_mask]]
            diff_low  = r_low  - r_l[j_idx[low_mask]]
            dist_high = np.linalg.norm(diff_high, axis=1)
            dist_low  = np.linalg.norm(diff_low,  axis=1)
            d_high_mean = float(dist_high.mean())
            d_low_mean  = float(dist_low.mean())
            delta = d_low_mean - d_high_mean
            # Cohen's d (pooled std)
            pooled_std = np.sqrt((dist_high.var() + dist_low.var()) / 2)
            cohens_d = delta / (pooled_std + 1e-12)
            # Mann-Whitney U
            u_stat, u_p = mannwhitneyu(dist_low, dist_high, alternative='greater')
            algo_res['layers'][f'l{l}'] = {
                'd_high_mean': d_high_mean,
                'd_low_mean':  d_low_mean,
                'delta':       delta,
                'cohens_d':    float(cohens_d),
                'mw_u':        float(u_stat),
                'mw_pvalue':   float(u_p),
                'dist_high_std': float(dist_high.std()),
                'dist_low_std':  float(dist_low.std()),
                'n_high': int(high_mask.sum()),
                'n_low':  int(low_mask.sum()),
            }
            print(f'  L{l}: D̄_high={d_high_mean:.4f}, D̄_low={d_low_mean:.4f}, '
                  f'Δ={delta:.4f}, d={cohens_d:.4f}, MW-p={u_p:.2e}')
        results[algo_name] = algo_res

    # Save
    out_path = os.path.join(OUT_DIR, 'idea1_phenomenon1.json')
    with open(out_path, 'w') as f_out:
        json.dump({
            'tau_high_pctl': TAU_HIGH_PCTL,
            'tau_low_pctl': TAU_LOW_PCTL,
            'n_pair_sample': N_PAIR_SAMPLE,
            'sim_cf_distribution': {
                'mean': float(sim_cf.mean()),
                'std': float(sim_cf.std()),
                'tau_high_abs': tau_high,
                'tau_low_abs': tau_low,
            },
            'results': results,
        }, f_out, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    # Verdict summary
    print('\n' + '=' * 70)
    print('VERDICT — Idea 1 现象 1')
    print('=' * 70)
    for algo, res in results.items():
        for lk, lr in res['layers'].items():
            sig = '✓' if lr['mw_pvalue'] < 0.01 and lr['cohens_d'] > 0.1 else '✗'
            direction = '+' if lr['delta'] > 0 else '-'
            print(f'  {algo} {lk}: Δ={direction}{lr["delta"]:.4f}, '
                  f'd={lr["cohens_d"]:.3f}, MW-p={lr["mw_pvalue"]:.2e}  {sig}')


if __name__ == '__main__':
    main()