#!/usr/bin/env python3
"""诊断组A 现象1 v2: 三空间持续同调对比 — 用自定义距离矩阵

修正 v1 错误: v1 把三个空间都做了相同 PCA-白化 + 欧氏 Rips, 导致结果完全一致.
正确做法:
  H0 persistence = single-linkage clustering 的 MST 边长 (sorted)
  - 用 scipy.cluster.hierarchy.linkage 接受自定义距离矩阵
  - X_raw / X_AQ: 欧氏距离 (Europ_white 化后)
  - X_HRQ:  Lorentz 距离 d_L(h1, h2) = arccosh(-<h1,h2>_L / c), 在 Poincaré ball 上做等价投影

数学:
  X_HRQ = {h_i ∈ R^(D+1) : h_i = [sqrt(c+||x_i||²), x_i]},  Lorentz model, c=1
  d_L(h1, h2) = arccosh( -<h1, h2>_L / c ),  <h1, h2>_L = -h1[0]*h2[0] + Σ h1[1:]·h2[1:]

  H0 死亡时刻 = single-linkage 树中所有边长 (sorted ascending)
  gap statistic: g_k = d_(k+1) - d_(k), g_max / g_second

复用: 与 v1 相同的 Stage 1 embedding, 同样的 3 seed.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json, time
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_idea5_hrq_contradiction'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 5000
PCA_DIM = 50
TOP_FRAC = 0.10
SEEDS = [42, 43, 44]
CURVATURE = 1.0


# ====================== Lorentz utilities ======================

def lift_to_lorentz(x, c=CURVATURE):
    x_norm2 = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)
    return torch.cat([h0, x], dim=-1)


def lorentz_pairwise_distance(H):
    """H: (N, D+1) tensor. Returns (N, N) pairwise distance matrix."""
    h0 = H[:, 0:1]                          # (N, 1)
    h1 = H[:, 1:]                           # (N, D)
    # <h_i, h_j>_L = -h0_i h0_j + Σ h1_i · h1_j
    inner = -(h0 @ h0.T) + h1 @ h1.T        # (N, N)
    arg = -inner / CURVATURE
    arg = torch.clamp(arg, min=1.0 + 1e-9)
    return torch.acosh(arg)                 # (N, N)


# ====================== White PCA (Euclidean) ======================

def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    X_pca = X_centered @ V
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    return X_white


# ====================== H0 from linkage ======================

def h0_deaths_from_distance_matrix(D_condensed):
    """scipy linkage(method='single') with precomputed condensed distances
       → returns (N-1, 4) linkage matrix
       → column 2 is the merge distance = edge length = H0 death time
       Returns: sorted ascending finite deaths array
    """
    Z = linkage(D_condensed, method='single')
    # Z[:, 2] are the merge distances = H0 deaths
    deaths = Z[:, 2]
    deaths = np.sort(deaths)
    deaths = deaths[np.isfinite(deaths)]
    return deaths


def gap_statistic(deaths, top_frac=TOP_FRAC):
    d = np.sort(deaths)
    n = len(d)
    n_keep = int(n * (1 - top_frac))
    d_keep = d[:n_keep]
    if len(d_keep) < 3:
        return {'g_max': float('nan'), 'g_second': float('nan'),
                'dominance_ratio': float('nan'), 'n_deaths_used': len(d_keep),
                'n_deaths_total': n}
    gaps = np.diff(d_keep)
    idx_max = int(np.argmax(gaps))
    g_max = float(gaps[idx_max])
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


# ====================== Space runners ======================

def run_euclidean_space(name, x_eu_white, seed):
    """对白化后的欧氏输入算 pairwise distance + single-linkage H0."""
    rng = np.random.default_rng(seed)
    N_sub_full = x_eu_white.shape[0]
    perm = rng.choice(N_sub_full, size=min(N_SUBSAMPLE, N_sub_full), replace=False)
    X_sub = x_eu_white[perm]  # (N, D_white) numpy

    # Pairwise Euclidean distance matrix — 用 sklearn or torch.cdist
    X_t = torch.from_numpy(X_sub).float()
    D = torch.cdist(X_t, X_t).numpy()
    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=False)
    deaths = h0_deaths_from_distance_matrix(condensed)
    gap = gap_statistic(deaths)
    return gap, len(deaths)


def run_lorentz_space(name, x_lorentz, seed):
    """对 Lorentz-lifted 数据算 pairwise Lorentz distance + single-linkage H0."""
    rng = np.random.default_rng(seed)
    N_sub_full = x_lorentz.shape[0]
    perm = rng.choice(N_sub_full, size=min(N_SUBSAMPLE, N_sub_full), replace=False)
    H_sub = x_lorentz[perm]  # (N, D+1) tensor

    D = lorentz_pairwise_distance(H_sub).numpy()
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    deaths = h0_deaths_from_distance_matrix(condensed)
    gap = gap_statistic(deaths)
    return gap, len(deaths)


# ====================== Main ======================

def main():
    print('=' * 70)
    print('诊断组A 现象1 v2: 三空间持续同调对比 (自定义距离矩阵版)')
    print('=' * 70)

    print('\n[Step 1] 加载 Stage 1 embedding')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N_full, D_eu = x_eu.shape
    print(f'  X_raw shape: {tuple(x_eu.shape)}')

    # X_raw / X_AQ pipeline: PCA-50 whiten
    print('\n[Step 2] 计算 PCA-50 白化 (用于 X_raw / X_AQ 欧氏距离)')
    x_white = whiten_pca(x_eu, n_components=PCA_DIM).numpy()
    print(f'  After PCA-50 whitening: {x_white.shape}')

    # X_HRQ pipeline: Lorentz 提升
    print('\n[Step 3] 提升到 Lorentz (用于 X_HRQ 双曲距离)')
    x_lorentz = lift_to_lorentz(x_eu).float()
    print(f'  X_HRQ shape: {tuple(x_lorentz.shape)}')

    spaces_results = {}

    # ---- X_raw ----
    print('\n[Step 4] X_raw: 欧氏距离 + PCA-50 白化')
    x_raw_seeds = {}
    for seed in SEEDS:
        t0 = time.time()
        gap, n_deaths = run_euclidean_space('X_raw', x_white, seed)
        elapsed = time.time() - t0
        print(f'    seed {seed}: dom_ratio={gap["dominance_ratio"]:.3f}, '
              f'n_deaths={n_deaths}, time={elapsed:.1f}s')
        x_raw_seeds[seed] = {'gap_stats': gap, 'n_deaths': n_deaths, 'time_sec': elapsed}
    spaces_results['X_raw'] = x_raw_seeds

    # ---- X_AQ (≈ X_raw; 操作于相同输入空间) ----
    print('\n[Step 5] X_AQ: 与 X_raw 输入相同 (AQ 不改变输入空间)')
    x_aq_seeds = {}
    for seed in SEEDS:
        t0 = time.time()
        gap, n_deaths = run_euclidean_space('X_AQ', x_white, seed)
        elapsed = time.time() - t0
        print(f'    seed {seed}: dom_ratio={gap["dominance_ratio"]:.3f}, '
              f'n_deaths={n_deaths}, time={elapsed:.1f}s')
        x_aq_seeds[seed] = {'gap_stats': gap, 'n_deaths': n_deaths, 'time_sec': elapsed}
    spaces_results['X_AQ'] = x_aq_seeds

    # ---- X_HRQ ----
    print('\n[Step 6] X_HRQ: Lorentz 距离')
    x_hrq_seeds = {}
    for seed in SEEDS:
        t0 = time.time()
        gap, n_deaths = run_lorentz_space('X_HRQ', x_lorentz, seed)
        elapsed = time.time() - t0
        print(f'    seed {seed}: dom_ratio={gap["dominance_ratio"]:.3f}, '
              f'n_deaths={n_deaths}, time={elapsed:.1f}s')
        x_hrq_seeds[seed] = {'gap_stats': gap, 'n_deaths': n_deaths, 'time_sec': elapsed}
    spaces_results['X_HRQ'] = x_hrq_seeds

    # ====================== 三空间汇总 ======================
    print('\n[Step 7] 三空间汇总')
    summary = {}
    for name, results in spaces_results.items():
        ratios = [r['gap_stats']['dominance_ratio'] for r in results.values()]
        summary[name] = {
            'ratios_per_seed': ratios,
            'mean': float(np.mean(ratios)),
            'std': float(np.std(ratios)),
            'min': float(min(ratios)),
            'max': float(max(ratios)),
        }
        print(f'  {name}: ratios={[round(r, 2) for r in ratios]}, '
              f'mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}')

    raw_mean = summary['X_raw']['mean']
    hrq_mean = summary['X_HRQ']['mean']
    aq_mean = summary['X_AQ']['mean']

    # AQ vs raw 应当高度一致 (AQ 操作于相同输入空间)
    aq_consistency = abs(aq_mean - raw_mean) / max(raw_mean, 1e-6)
    print(f'\n  AQ vs raw 一致性: |Δ|/raw = {aq_consistency:.2%}')

    # 三结局分流
    if hrq_mean > raw_mean * 1.15:
        outcome = (
            f'(a) HRQ 训练拉出更多可分辨尺度: '
            f'X_raw={raw_mean:.2f} → X_HRQ={hrq_mean:.2f} '
            f'(提升 {(hrq_mean/raw_mean - 1)*100:.1f}%). '
            f'矛盾调和: HRQ 价值是"主动诱导多尺度结构", 而非被动匹配数据本有的树形. '
            f'需要重写 HRQ 的 motivation 段落.'
        )
        outcome_letter = 'a'
    elif hrq_mean < raw_mean * 0.85:
        outcome = (
            f'(c) HRQ 反压平结构: '
            f'X_raw={raw_mean:.2f} → X_HRQ={hrq_mean:.2f} '
            f'(下降 {(1 - hrq_mean/raw_mean)*100:.1f}%). '
            f'需要重新审视 +37% R@10 是否真来自"双曲几何"机制.'
        )
        outcome_letter = 'c'
    else:
        outcome = (
            f'(b) HRQ 与 raw 结构相似: '
            f'X_raw={raw_mean:.2f} ≈ X_HRQ={hrq_mean:.2f} '
            f'(差距 {(hrq_mean-raw_mean)/raw_mean*100:.1f}%, 在 ±15% 内). '
            f'结论: 双曲空间的几何收益**不来自"层级匹配"**, 必须另寻机制. '
            f'自然后续: 走 FDR (Fisher 判别比) 检验"边界可分性"机制.'
        )
        outcome_letter = 'b'

    print(f'\n  Final outcome: {outcome}')

    verdict = (
        f'OUTCOME_{outcome_letter.upper()}: {outcome}\n'
        f'AQ consistency with raw: {aq_consistency:.1%}'
    )
    print(f'\n[Verdict] {verdict}')

    out_json = os.path.join(OUT_DIR, 'three_space_persistence_v2.json')
    with open(out_json, 'w') as f:
        json.dump({
            'description': (
                'Vietoris-Rips H0 via single-linkage clustering on custom distance matrices. '
                'X_raw/X_AQ: Euclidean on PCA-50 whitened embedding. '
                'X_HRQ: Lorentz distance on Lorentz-lifted data.'
            ),
            'params': {
                'n_subsample': N_SUBSAMPLE,
                'pca_dim': PCA_DIM,
                'top_frac_excluded': TOP_FRAC,
                'curvature': CURVATURE,
                'seeds': SEEDS,
            },
            'spaces_summary': summary,
            'spaces_results': {
                name: {str(s): {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                 for k, v in r.items()}
                       for s, r in results.items()}
                for name, results in spaces_results.items()
            },
            'comparisons': {
                'raw_to_hrq_ratio': float(hrq_mean / raw_mean) if raw_mean > 0 else float('nan'),
                'aq_consistency_with_raw': float(aq_consistency),
            },
            'outcome': outcome,
            'outcome_letter': outcome_letter,
            'verdict': verdict,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')
    return verdict, summary


if __name__ == '__main__':
    main()
