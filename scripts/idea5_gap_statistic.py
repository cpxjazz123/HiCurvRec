#!/usr/bin/env python3
"""Idea 5 诊断组B 现象1: gap statistic 替代 KDE 峰计数

目的:
- 给"主导尺度"一个不依赖 KDE 带宽选择的客观判据
- 跨 3 个种子 (42/43/44) 独立验证

定义:
- d_(1) ≤ d_(2) ≤ ... ≤ d_(n): H0 死亡时刻排序 (排除最后 10% 大 connected component 边界)
- g_k = d_(k+1) - d_(k), k=1,...,n-1
- g_max = max(g_k)
- g_second = max_{k != argmax} g_k
- dominance_ratio = g_max / g_second

判定:
- 3 seed dominance_ratio > 3 → ROBUST_SINGLE_SCALE (主峰确实主导)
- 高方差 / 比值不稳定 → INSUFFICIENT_DISCRIMINATION
- 比值在 2-8 区间波动 → CAUTION

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
- 已用到的 idea5_tda_supplementary.py 的子采样和 PCA-白化协议
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
import ripser

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea5_tda'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 5000
PCA_DIM = 50
RIPSER_THRESHOLD_PERCENTILE = 90
TOP_FRAC = 0.10  # 排除最后 10% 的边界大 gap


def whiten_pca(X, n_components=50):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    X_pca = X_centered @ V
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    return X_white


def compute_h0(X_np, thresh):
    result = ripser.ripser(X_np, thresh=thresh)['dgms']
    h0 = result[0]
    h0 = h0[np.isfinite(h0[:, 1])]
    return h0


def gap_statistic(deaths, top_frac=TOP_FRAC):
    """对死亡时刻排序, 排除末尾 top_frac 的边界, 找 max / second-max gap.
       Returns: dict with g_max, g_second, dominance_ratio, gap_position
    """
    d = np.sort(deaths)
    n = len(d)
    # 排除末尾 top_frac (边界 component 必有大 gap)
    n_keep = int(n * (1 - top_frac))
    d_keep = d[:n_keep]

    if len(d_keep) < 3:
        return {
            'g_max': float('nan'),
            'g_second': float('nan'),
            'dominance_ratio': float('nan'),
            'g_max_position': -1,
            'g_max_at_death': float('nan'),
            'n_deaths_used': len(d_keep),
            'n_deaths_total': n,
        }

    gaps = np.diff(d_keep)
    # First 最大的
    idx_max = int(np.argmax(gaps))
    g_max = float(gaps[idx_max])
    # Second 最大的 = 排序后的第二大
    sorted_gaps = np.sort(gaps)[::-1]
    g_second = float(sorted_gaps[1]) if len(sorted_gaps) >= 2 else float('nan')
    dominance_ratio = float(g_max / g_second) if g_second > 1e-12 else float('inf')

    return {
        'g_max': g_max,
        'g_second': g_second,
        'dominance_ratio': dominance_ratio,
        'g_max_position': idx_max,
        'g_max_at_death': float(d_keep[idx_max]),
        'n_deaths_used': len(d_keep),
        'n_deaths_total': n,
    }


def main():
    print('=' * 70)
    print('Idea 5 诊断组B 现象1: gap statistic 替代 KDE 峰计数')
    print('=' * 70)

    print('\n[Step 1] 加载 Stage 1 embedding')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N_full, d = emb.shape
    print(f'  Full embedding: {tuple(emb.shape)}')

    seeds = [42, 43, 44]
    results_per_seed = {}

    for seed in seeds:
        print(f'\n[seed = {seed}]')
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N_full, generator=g)[:N_SUBSAMPLE]
        emb_sub = emb[perm]

        emb_white = whiten_pca(emb_sub, n_components=PCA_DIM)
        X_np = emb_white.numpy()

        rng = np.random.default_rng(seed)
        n_dist_sample = 50000
        i_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
        j_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        dists = np.linalg.norm(X_np[i_idx] - X_np[j_idx], axis=1)

        thresh = float(np.percentile(dists, RIPSER_THRESHOLD_PERCENTILE))
        h0 = compute_h0(X_np, thresh=thresh)
        deaths = h0[:, 1]

        gap_stats = gap_statistic(deaths, top_frac=TOP_FRAC)

        print(f'  thresh={thresh:.4f}, n_pairs={len(h0)}, n_deaths(过滤边界前)={gap_stats["n_deaths_used"]}/{gap_stats["n_deaths_total"]}')
        print(f'  g_max = {gap_stats["g_max"]:.5f} at death = {gap_stats["g_max_at_death"]:.4f}')
        print(f'  g_second = {gap_stats["g_second"]:.5f}')
        print(f'  dominance_ratio = {gap_stats["dominance_ratio"]:.3f}')

        results_per_seed[seed] = {
            'gap_stats': gap_stats,
            'n_pairs': len(h0),
            'thresh': thresh,
            'death_min': float(deaths.min()),
            'death_max': float(deaths.max()),
            'death_median': float(np.median(deaths)),
            'death_std': float(deaths.std()),
        }

    # 判定
    ratios = [r['gap_stats']['dominance_ratio'] for r in results_per_seed.values()]
    print(f'\n[Step 2] 跨 3 seed 汇总 dominance_ratio: {[round(r, 3) for r in ratios]}')
    ratio_mean = float(np.mean(ratios))
    ratio_std = float(np.std(ratios))
    ratio_min = float(min(ratios))
    ratio_max = float(max(ratios))
    ratio_range = ratio_max - ratio_min
    print(f'  均值={ratio_mean:.3f}, 标准差={ratio_std:.3f}')
    print(f'  区间 [{ratio_min:.3f}, {ratio_max:.3f}], 跨度={ratio_range:.3f}')

    # 判定逻辑
    all_pass_3 = all(r > 3 for r in ratios)
    all_pass_2 = all(r > 2 for r in ratios)
    stable_low = ratio_std < 1.0 and ratio_range < 3.0

    if all_pass_3 and stable_low:
        verdict = (
            f'ROBUST_SINGLE_SCALE: 3 seed dominance_ratio 全部 > 3 '
            f'({[round(r, 2) for r in ratios]}), 标准差 {ratio_std:.2f}, '
            f'主峰客观存在且稳定地主导, 可以写入正式文档'
        )
    elif all_pass_2 and stable_low:
        verdict = (
            f'ROBUST_SINGLE_SCALE_WEAK: 3 seed 全部 > 2 但不全部 > 3 '
            f'({[round(r, 2) for r in ratios]}), 主峰主导性存在但不算强, '
            f'需如实标注"主要合并事件 + 次峰邻近"'
        )
    elif ratio_range > 3.0:
        verdict = (
            f'INSUFFICIENT_DISCRIMINATION: dominance_ratio 跨种子跨度 {ratio_range:.2f} '
            f'({[round(r, 2) for r in ratios]}), 数据本身尺度结构处于模糊地带, '
            f'此诊断法在当前数据规模下判别力不足'
        )
    else:
        verdict = (
            f'CAUTION: dominance_ratio 均值 {ratio_mean:.2f} 但跨越 "单/多尺度" 边界, '
            f'({[round(r, 2) for r in ratios]}) 需更细颗粒度的后续诊断'
        )

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea5_gap_statistic.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'gap_statistic',
            'description': (
                'H0 死亡时刻排序, 找 max/second-max gap, '
                'dominance_ratio = g_max / g_second'
            ),
            'params': {
                'n_subsample': N_SUBSAMPLE,
                'pca_dim': PCA_DIM,
                'thresh_percentile': RIPSER_THRESHOLD_PERCENTILE,
                'top_frac_excluded': TOP_FRAC,
            },
            'results_per_seed': results_per_seed,
            'summary': {
                'ratios': [float(r) for r in ratios],
                'mean': ratio_mean,
                'std': ratio_std,
                'min': ratio_min,
                'max': ratio_max,
                'range': ratio_range,
                'stable_low': stable_low,
                'all_pass_3': all_pass_3,
                'all_pass_2': all_pass_2,
            },
            'verdict': verdict,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task10_persistent_homology_tda.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    return verdict, results_per_seed


if __name__ == '__main__':
    main()
