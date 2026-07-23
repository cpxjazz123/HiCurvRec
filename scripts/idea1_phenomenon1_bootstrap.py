#!/usr/bin/env python3
"""Bootstrap CI for Δ_l — Idea 1 现象 1 强化版"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json, numpy as np, torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'

CF_PATH = os.path.join(OUT_DIR, 'cf_ppmi_svd256.pt')
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}

N_BOOT = 500
N_BOOT_SAMPLE = 1500   # pairs per bootstrap iteration (subsample for speed)


def bootstrap_ci(delta_arr, n_boot=N_BOOT, ci=0.95):
    """Mean and CI of an array of per-iteration Δ values."""
    lo = float(np.quantile(delta_arr, (1-ci)/2))
    hi = float(np.quantile(delta_arr, 1 - (1-ci)/2))
    return lo, hi


def main():
    print('Loading PPMI f_i')
    f = torch.load(CF_PATH, map_location='cpu', weights_only=False)['f_normalized'].numpy()
    N = f.shape[0]
    rng = np.random.default_rng(42)

    # Sample large pair pool once
    N_PAIR_POOL = 60000
    print(f'Sampling {N_PAIR_POOL} pairs...')
    i_idx = rng.integers(0, N, size=N_PAIR_POOL)
    j_idx = rng.integers(0, N, size=N_PAIR_POOL)
    valid = i_idx != j_idx
    i_idx, j_idx = i_idx[valid], j_idx[valid]
    sim_cf = (f[i_idx] * f[j_idx]).sum(axis=1)

    tau_high = float(np.quantile(sim_cf, 0.90))
    tau_low  = float(np.quantile(sim_cf, 0.10))
    high_mask = sim_cf > tau_high
    low_mask  = sim_cf < tau_low
    high_i, high_j = i_idx[high_mask], j_idx[high_mask]
    low_i,  low_j  = i_idx[low_mask],  j_idx[low_mask]
    print(f'  |P_high| = {len(high_i)}, |P_low| = {len(low_i)}')

    print('\nLoading r_lst for A/B/C...')
    bundles = {}
    for name, p in RQIDX_PATHS.items():
        bundles[name] = torch.load(p, map_location='cpu', weights_only=False)

    results = {}
    for algo_name, bundle in bundles.items():
        print(f'\n--- {algo_name} ---')
        r_lst = bundle['r_lst']
        L = len(r_lst) - 1
        algo_res = {}
        for l in range(1, L+1):
            r_l = r_lst[l].numpy()
            # Pre-fetch high/low diff vectors
            diff_high_all = r_l[high_i] - r_l[high_j]
            diff_low_all  = r_l[low_i]  - r_l[low_j]
            dist_high_all = np.linalg.norm(diff_high_all, axis=1)
            dist_low_all  = np.linalg.norm(diff_low_all,  axis=1)

            # Bootstrap: subsample pairs, compute Δ
            n_h, n_l = len(high_i), len(low_i)
            deltas = np.empty(N_BOOT)
            for b in range(N_BOOT):
                hidx = rng.integers(0, n_h, size=N_BOOT_SAMPLE)
                lidx = rng.integers(0, n_l, size=N_BOOT_SAMPLE)
                dh = dist_high_all[hidx].mean()
                dl = dist_low_all[lidx].mean()
                deltas[b] = dl - dh
            d_mean = float(deltas.mean())
            d_lo, d_hi = bootstrap_ci(deltas)
            # Effect size from full sample
            full_delta = dist_low_all.mean() - dist_high_all.mean()
            pooled = np.sqrt((dist_high_all.var() + dist_low_all.var()) / 2)
            cohens_d = full_delta / (pooled + 1e-12)

            algo_res[f'l{l}'] = {
                'delta_mean': d_mean,
                'delta_ci_low': d_lo,
                'delta_ci_high': d_hi,
                'delta_full': float(full_delta),
                'cohens_d': float(cohens_d),
                'ci_excludes_zero': bool(d_lo > 0),
            }
            sign = '+' if d_mean > 0 else '-'
            ci_sig = 'CI>0' if d_lo > 0 else 'CI⊃0'
            d_sig = '✓d>0.1' if cohens_d > 0.1 else '✗d<0.1'
            print(f'  L{l}: Δ={sign}{d_mean:.4f} [{d_lo:.4f}, {d_hi:.4f}], '
                  f'd={cohens_d:.3f}  {ci_sig} {d_sig}')
        results[algo_name] = algo_res

    out_path = os.path.join(OUT_DIR, 'idea1_phenomenon1_bootstrap.json')
    with open(out_path, 'w') as f_out:
        json.dump({
            'n_boot': N_BOOT,
            'n_boot_sample': N_BOOT_SAMPLE,
            'tau_high': tau_high,
            'tau_low': tau_low,
            'results': results,
        }, f_out, indent=2)
    print(f'\n=== Saved → {out_path} ===')


if __name__ == '__main__':
    main()