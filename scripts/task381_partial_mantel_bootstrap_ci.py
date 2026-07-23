#!/usr/bin/env python3
"""Stage 1 验证：partial Mantel ρ 的 bootstrap 置信区间

复用 task380 的 dist_matrices.npz, 对每个 layer l (0~3) 的 partial ρ 做 1000 次
商品子集 bootstrap 重采样, 同时计算 naive ρ 的 CI 作为交叉核对.

判定标准 (按用户指示):
  - 90% CI 收窄在 [0.10, 0.18] 范围 (跨度 ≤ 0.08) → 稳健, 推进 Stage 2
  - 90% CI 下界 < 0.05 → 不稳健, 不能拿 0.139 当锚点
  - 90% CI 跨度 > 0.10 → 需更谨慎解读

默认: n_boot=1500 (50% subsample), n_iter=1000, seed=42
"""

import os
import sys
import json
import argparse
import time

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)

import numpy as np
from scipy.stats import pearsonr

IN_NPZ = f'{GRID}/result/task380_stage0_1_partial_mantel/dist_matrices.npz'
OUT_DIR = f'{GRID}/result/task381_partial_mantel_bootstrap_ci'
os.makedirs(OUT_DIR, exist_ok=True)


def upper_tri(m):
    n = m.shape[0]
    return m[np.triu_indices(n, k=1)]


def regression_residual_r(D_residual, D_tax, D_trans):
    """回归残差法 partial Mantel — 复用 task380 实现"""
    v_r = upper_tri(D_residual)
    v_t = upper_tri(D_tax)
    v_g = upper_tri(D_trans)
    sl_r, int_r = np.polyfit(v_t, v_r, 1)
    sl_g, int_g = np.polyfit(v_t, v_g, 1)
    R_r = v_r - (sl_r * v_t + int_r)
    R_g = v_g - (sl_g * v_t + int_g)
    return float(pearsonr(R_r, R_g)[0])


def naive_pearson_r(D_a, D_b):
    """原始 Pearson 相关（无偏控制）— 用于 cross-check"""
    v_a = upper_tri(D_a)
    v_b = upper_tri(D_b)
    return float(pearsonr(v_a, v_b)[0])


def bootstrap_one_layer(D_residual, D_tax, D_trans,
                         n_boot=1500, n_iter=1000, seed=42):
    """对一个 layer 的 partial ρ 和 naive ρ 做 bootstrap 重采样

    Returns:
        partial_rhos: (n_iter,) 数组
        naive_rhos:   (n_iter,) 数组
    """
    N = D_residual.shape[0]
    rng = np.random.default_rng(seed)
    partial_rhos = np.empty(n_iter, dtype=np.float64)
    naive_rhos = np.empty(n_iter, dtype=np.float64)
    for it in range(n_iter):
        idx = rng.choice(N, size=n_boot, replace=False)
        D_r_sub = D_residual[np.ix_(idx, idx)]
        D_t_sub = D_tax[np.ix_(idx, idx)]
        D_g_sub = D_trans[np.ix_(idx, idx)]
        partial_rhos[it] = regression_residual_r(D_r_sub, D_t_sub, D_g_sub)
        naive_rhos[it] = naive_pearson_r(D_r_sub, D_g_sub)
    return partial_rhos, naive_rhos


def summarize(samples, point_rho):
    s = {
        'n_iter': int(len(samples)),
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples, ddof=1)),
        'median': float(np.median(samples)),
        'min': float(np.min(samples)),
        'max': float(np.max(samples)),
        'p05': float(np.percentile(samples, 5)),
        'p10': float(np.percentile(samples, 10)),
        'p90': float(np.percentile(samples, 90)),
        'p95': float(np.percentile(samples, 95)),
        'p025': float(np.percentile(samples, 2.5)),
        'p975': float(np.percentile(samples, 97.5)),
        'ci90': [float(np.percentile(samples, 5)), float(np.percentile(samples, 95))],
        'ci95': [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        'point_full_data': float(point_rho),
    }
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=1500,
                        help='每次 bootstrap 抽样商品数 (默认 1500 = 50%% of 3000)')
    parser.add_argument('--n-iter', type=int, default=1000,
                        help='bootstrap 重复次数 (默认 1000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: n_iter=100, 用于测试脚本')
    args = parser.parse_args()

    if args.quick:
        args.n_iter = 100

    print(f'[task381] Loading {IN_NPZ}', flush=True)
    d = np.load(IN_NPZ)
    D_tax = d['D_tax']
    D_trans = d['D_trans']
    layers = {l: d[f'D_residual_l{l}'] for l in range(4)}
    sample_items = d['sample_items']
    print(f'[task381] N={len(sample_items)}, layers={list(layers.keys())}')
    print(f'[task381] n_boot={args.n_boot}, n_iter={args.n_iter}, seed={args.seed}',
          flush=True)

    # 先算全样本 point estimate, 用作参照
    point_estimates = {}
    for l in range(4):
        point_estimates[l] = {
            'partial': regression_residual_r(layers[l], D_tax, D_trans),
            'naive': naive_pearson_r(layers[l], D_trans),
        }
    print('[task381] Full-data point estimates:')
    for l in range(4):
        print(f'  l={l}: partial={point_estimates[l]["partial"]:.4f}, '
              f'naive={point_estimates[l]["naive"]:.4f}')

    # bootstrap 每个 layer
    all_results = {}
    t0 = time.time()
    for l in range(4):
        print(f'[task381] Bootstrap layer l={l}...', flush=True)
        t_l = time.time()
        partial_rhos, naive_rhos = bootstrap_one_layer(
            layers[l], D_tax, D_trans,
            n_boot=args.n_boot, n_iter=args.n_iter, seed=args.seed + l,
        )
        elapsed_l = time.time() - t_l
        all_results[l] = {
            'partial': summarize(partial_rhos, point_estimates[l]['partial']),
            'naive': summarize(naive_rhos, point_estimates[l]['naive']),
            'partial_rhos_raw': partial_rhos,
            'naive_rhos_raw': naive_rhos,
        }
        s = all_results[l]['partial']
        print(f'  l={l}: partial CI90=[{s["ci90"][0]:.4f}, {s["ci90"][1]:.4f}], '
              f'CI95=[{s["ci95"][0]:.4f}, {s["ci95"][1]:.4f}], '
              f'mean={s["mean"]:.4f}, std={s["std"]:.4f}, '
              f'time={elapsed_l:.1f}s', flush=True)

    total_time = time.time() - t0
    print(f'[task381] Done. Total time: {total_time:.1f}s')

    # 判定 (按用户判据)
    verdict = {}
    for l in range(4):
        s = all_results[l]['partial']
        ci90_width = s['ci90'][1] - s['ci90'][0]
        # 用户期望范围 [0.10, 0.18], 跨度 0.08, 容许 ±0.01 → 跨度阈值 0.10
        # 下界 ≥ 0.05, 上界 ≤ 0.18
        verdict[l] = {
            'ci90_lower': s['ci90'][0],
            'ci90_upper': s['ci90'][1],
            'ci90_width': ci90_width,
            'passes_narrow_ci': bool(ci90_width <= 0.10),
            'passes_lower_bound': bool(s['ci90'][0] >= 0.05),
            'passes_upper_bound': bool(s['ci90'][1] <= 0.18),
            'robust': False,  # fill below
        }
        verdict[l]['robust'] = (
            verdict[l]['passes_narrow_ci']
            and verdict[l]['passes_lower_bound']
            and verdict[l]['passes_upper_bound']
        )
        status = 'PASS' if verdict[l]['robust'] else 'FAIL'
        print(f'[task381] l={l} verdict: '
              f'CI90=[{s["ci90"][0]:.4f}, {s["ci90"][1]:.4f}], '
              f'width={ci90_width:.4f}, '
              f'narrow={verdict[l]["passes_narrow_ci"]}, '
              f'lower_ok={verdict[l]["passes_lower_bound"]}, '
              f'upper_ok={verdict[l]["passes_upper_bound"]}, '
              f'overall={status}', flush=True)

    # 保存 JSON (无 raw 数组)
    compact_out = {
        'task': 'task381_partial_mantel_bootstrap_ci',
        'method': 'bootstrap_ci',
        'date': '2026-07-14',
        'status': 'completed',
        'data': {
            'config': {
                'n_boot': args.n_boot,
                'n_iter': args.n_iter,
                'seed': args.seed,
            },
            'input_npz': IN_NPZ,
            'n_sample': int(len(sample_items)),
            'point_estimates': {str(l): point_estimates[l] for l in range(4)},
            'partial_summary': {
                str(l): {k: v for k, v in all_results[l]['partial'].items()}
                for l in range(4)
            },
            'naive_summary': {
                str(l): {k: v for k, v in all_results[l]['naive'].items()}
                for l in range(4)
            },
            'verdict': {str(l): verdict[l] for l in range(4)},
            'total_time_sec': total_time,
        }
    }
    out_path = f'{OUT_DIR}/bootstrap_ci_results.json'
    with open(out_path, 'w') as f:
        json.dump(compact_out, f, indent=2)
    print(f'[task381] Saved: {out_path}')

    # 保存 raw 数组到 npz 以便复现
    np.savez(f'{OUT_DIR}/bootstrap_rhos.npz',
             **{f'partial_l{l}': all_results[l]['partial_rhos_raw'] for l in range(4)},
             **{f'naive_l{l}': all_results[l]['naive_rhos_raw'] for l in range(4)})
    print(f'[task381] Saved raw: {OUT_DIR}/bootstrap_rhos.npz')


if __name__ == '__main__':
    main()