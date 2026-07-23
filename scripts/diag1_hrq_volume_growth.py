#!/usr/bin/env python3
"""HRQ 诊断1: 体积增长指数 — 直接测"空间是否真的在指数展开容量"

目的: 不问"数据像不像树", 直接问"空间的容量增长速度, 到底是不是指数级"

数学:
- 欧氏空间: N(r) ∝ r^k (k 是有效维度), 即 log N(r) vs log r 线性
- 双曲空间: N(r) ∝ e^{(d-1)√-κ · r}, 即 log N(r) vs r 线性

做法:
- 对每个 item i, 取中心, 数半径 r 内的邻居数
- 计算 log N(r) vs r 的 R² (双曲假设)
- 计算 log N(r) vs log r 的 R² (欧氏假设)
- 比较两种拟合的优劣, 在 X_raw 和 X_HRQ 两个空间都做

判定:
- 若 X_HRQ 上 log N(r) vs r 的 R² 显著优于 log N(r) vs log r,
  且 X_raw 正好相反 → HRQ 训练确实把空间"整形"成了指数增长结构
- 若两空间增长模式无本质差异 → KILL: 双曲几何没有改变容量增长模式,
  +37% 收益必然来自别的机制

复用:
- Stage 1 flan-t5-xl embedding (X_raw)
- task20 的 lift_to_lorentz + lorentz_distance (X_HRQ)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag1_hrq_volume_growth'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_CENTERS = 200      # 取 200 个 item 作为邻居计数中心
K_NEIGHBORS_MAX = 500  # 每个中心看最大 500 个邻居
N_R_SAMPLES = 30     # 半径采样点数
PCA_DIM = 50         # 白化维度
SEED = 42


def lift_to_lorentz(x, c=1.0):
    x_norm2 = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)
    return torch.cat([h0, x], dim=-1)


def lorentz_distance_matrix(H):
    """Compute (N, N) pairwise Lorentz distance."""
    h0 = H[:, 0:1]
    h1 = H[:, 1:]
    inner = -(h0 @ h0.T) + h1 @ h1.T
    arg = torch.clamp(-inner / 1.0, min=1.0 + 1e-9)
    return torch.acosh(arg)


def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    return (X_centered @ V) / (S[:n_components].clamp(min=1e-8))


def growth_curve_analysis(dists_for_centers, r_grid):
    """对每个中心, 数 N(r), 然后取所有中心的 N(r) 均值.
       Returns: N_mean (R_grid_length,)
    """
    # dists_for_centers: (N_centers, K), r_grid: (R,)
    # counts: (N_centers, R) — for each center, number of neighbors within r
    counts = (dists_for_centers[:, :, None] < r_grid[None, None, :]).sum(axis=1)  # (N_centers, R)
    N_mean = counts.mean(axis=0).astype(float)  # (R,)
    return N_mean


def linear_fit_score(xs, ys):
    """Compute R² of linear regression on log N vs r (hyperbolic hypothesis)
       Or log N vs log r (Euclidean hypothesis)"""
    from scipy.stats import linregress
    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    if mask.sum() < 3:
        return {'slope': float('nan'), 'r_squared': float('nan'),
                'intercept': float('nan'), 'n_used': int(mask.sum())}
    res = linregress(xs[mask], np.log(ys[mask]))
    # R² vs log-ys (since we already took log)
    return {
        'slope': float(res.slope),
        'intercept': float(res.intercept),
        'r_squared': float(res.rvalue ** 2),
        'p_value': float(res.pvalue),
        'n_used': int(mask.sum()),
    }


def main():
    print('=' * 70)
    print('HRQ 诊断1: 体积增长指数 — log N(r) vs r vs log r')
    print('=' * 70)

    # 加载 X_raw
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N_full, D_eu = x_eu.shape
    print(f'\n  X_raw shape: {tuple(x_eu.shape)}')

    # PCA 白化 + Lorentz 提升
    rng = np.random.default_rng(SEED)
    x_white = whiten_pca(x_eu, n_components=PCA_DIM).numpy()  # (N, 50)
    print(f'  After PCA-{PCA_DIM} whitening: {x_white.shape}')

    x_lorentz = lift_to_lorentz(x_eu).float()  # (N, 2049)
    print(f'  X_HRQ (Lorentz) shape: {tuple(x_lorentz.shape)}')

    # 选 200 个中心 + 500 个邻居 (距离矩阵只算子集, 加速)
    center_idx = rng.choice(N_full, size=N_CENTERS, replace=False)
    neighbor_idx = rng.choice(N_full, size=K_NEIGHBORS_MAX, replace=False)
    print(f'\n  Selected {N_CENTERS} centers, {K_NEIGHBORS_MAX} neighbors')

    # ===================== X_raw 距离 =====================
    print('\n[Step 1] X_raw pairwise Euclidean distance')
    x_white_t = torch.from_numpy(x_white)
    centers_w = x_white_t[center_idx]
    neighbors_w = x_white_t[neighbor_idx]
    D_raw = torch.cdist(centers_w, neighbors_w).numpy()  # (N_C, K)
    print(f'  Distance range: [{D_raw.min():.4f}, {D_raw.max():.4f}], '
          f'median={np.median(D_raw):.4f}')

    # ===================== X_HRQ 距离 =====================
    print('\n[Step 2] X_HRQ pairwise Lorentz distance')
    centers_l = x_lorentz[center_idx]
    neighbors_l = x_lorentz[neighbor_idx]
    h0_c = centers_l[:, 0:1]
    h1_c = centers_l[:, 1:]
    h0_n = neighbors_l[:, 0:1]
    h1_n = neighbors_l[:, 1:]
    inner = -(h0_c @ h0_n.T) + h1_c @ h1_n.T  # (N_C, K)
    arg = torch.clamp(-inner, min=1.0 + 1e-9)
    D_hrq = torch.acosh(arg).numpy()
    print(f'  Distance range: [{D_hrq.min():.4f}, {D_hrq.max():.4f}], '
          f'median={np.median(D_hrq):.4f}')

    # ===================== 拟合两种模型 =====================
    print('\n[Step 3] 拟合两种增长模型')

    # 对每个空间, 选 30 个 r 采样点 (用每个空间的距离分位数)
    raw_qs = np.linspace(D_raw.min() + 1e-9, np.percentile(D_raw, 99), N_R_SAMPLES)
    hrq_qs = np.linspace(D_hrq.min() + 1e-9, np.percentile(D_hrq, 99), N_R_SAMPLES)

    results = {}
    for name, D, r_grid in [
        ('X_raw', D_raw, raw_qs),
        ('X_HRQ', D_hrq, hrq_qs),
    ]:
        print(f'\n  === Space: {name} ===')
        N_mean = growth_curve_analysis(D, r_grid)
        log_N = np.log(N_mean + 1e-9)
        log_r = np.log(r_grid + 1e-9)

        # Hyperbolic 假设: log N vs r 线性
        fit_hyp = linear_fit_score(r_grid, N_mean)
        # Euclidean 假设: log N vs log r 线性
        fit_euc = linear_fit_score(log_r, N_mean)

        print(f'  r 范围: [{r_grid.min():.4f}, {r_grid.max():.4f}]')
        print(f'  N_mean 范围: [{N_mean.min():.1f}, {N_mean.max():.1f}]')
        print(f'    [Hyperbolic: log N vs r]      R² = {fit_hyp["r_squared"]:.4f}, '
              f'slope = {fit_hyp["slope"]:.4f}')
        print(f'    [Euclidean:  log N vs log r]  R² = {fit_euc["r_squared"]:.4f}, '
              f'slope = {fit_euc["slope"]:.4f}')

        winner = 'hyperbolic' if fit_hyp['r_squared'] > fit_euc['r_squared'] else 'euclidean'
        margin = abs(fit_hyp['r_squared'] - fit_euc['r_squared'])
        print(f'    Winner: {winner} (margin = {margin:.4f})')

        results[name] = {
            'r_grid': r_grid.tolist(),
            'N_mean': N_mean.tolist(),
            'hyp_fit': fit_hyp,
            'euc_fit': fit_euc,
            'winner': winner,
            'R2_margin': margin,
        }

    # ===================== 判定 =====================
    raw_winner = results['X_raw']['winner']
    hrq_winner = results['X_HRQ']['winner']
    raw_margin = results['X_raw']['R2_margin']
    hrq_margin = results['X_HRQ']['R2_margin']

    print(f'\n[Step 4] 判定')
    print(f'  X_raw 偏好: {raw_winner} (margin={raw_margin:.4f})')
    print(f'  X_HRQ 偏好: {hrq_winner} (margin={hrq_margin:.4f})')

    if raw_winner == 'euclidean' and hrq_winner == 'hyperbolic':
        verdict = (
            'CONFIRMED_HYPERBOLIC_GROWTH: X_raw 偏好欧氏幂律, X_HRQ 偏好双曲指数. '
            'HRQ 训练确实把空间整形成了指数增长结构, 双曲几何的核心假设在 HRQ 数据上验证成立. '
            '+37% R@10 收益有直接的几何依据.'
        )
        outcome = 'confirmed'
    elif hrq_winner == 'hyperbolic' and hrq_margin > 0.05:
        verdict = (
            'PARTIAL_HYPERBOLIC: 两空间都偏向双曲, 但 HRQ 偏向更显著. '
            '双曲性质比欧氏更显著, 但 X_raw 偏离欧氏是一个需要注意的混杂.'
        )
        outcome = 'partial'
    elif hrq_winner == raw_winner:
        verdict = (
            f'KILL_NO_GROWTH_DIFFERENCE: 两空间增长模式无本质差异 '
            f'(都偏 {raw_winner}, margin {raw_margin:.4f}/{hrq_margin:.4f}). '
            '双曲几何没有改变容量增长模式. +37% 收益必然来自别的机制.'
        )
        outcome = 'kill'
    else:
        verdict = (
            f'MIXED: X_raw 偏好 {raw_winner}, X_HRQ 偏好 {hrq_winner}. '
            f'结果不干净, 需要更多 seed 或更大子采样确认.'
        )
        outcome = 'mixed'

    print(f'\n[Verdict] {verdict}')

    out_json = os.path.join(OUT_DIR, 'volume_growth.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'volume_growth_exponent',
            'description': (
                'N(r) = |{j : d(x_i, x_j) < r}|. Compare R² of '
                'log N vs r (hyperbolic hypothesis) vs log N vs log r (Euclidean hypothesis).'
            ),
            'params': {
                'n_centers': N_CENTERS,
                'k_neighbors_max': K_NEIGHBORS_MAX,
                'n_r_samples': N_R_SAMPLES,
                'pca_dim': PCA_DIM,
                'seed': SEED,
            },
            'results_per_space': results,
            'verdict': verdict,
            'outcome': outcome,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')
    return verdict, results


if __name__ == '__main__':
    main()
