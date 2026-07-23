#!/usr/bin/env python3
"""Idea 5 TDA 现象 1: H0 持续图 + 死亡时刻 KDE 峰计数 n_peaks

测什么:
- 子采样 5000 item embedding (seed=42, 复用 task18 子采样)
- 白化预处理
- Vietoris-Rips filtration → H0 持续图 {(b=0, d_i)}
- 死亡时刻 d_i KDE → 数显著峰 n_peaks
- 持续熵 E_pers

n_peaks 判定:
- n_peaks = 1 → 数据层级 ≈ 1, 量化算法无责
- n_peaks ≥ 2 且尺度分离 → 层级真, 量化未对准, 开新方法线

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea5_tda'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 5000
SEED = 42
RIPSER_THRESHOLD = 2.0  # 距离上限, 防止太远合并
PCA_DIM = 50  # 白化前先降到 50 维, 否则 2048 维距离毫无意义


def whiten_pca(X, n_components=50):
    """PCA 白化: 投影到 n_components 维, 单位方差"""
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # 取 top n_components
    V = Vh[:n_components].T  # (d, n_components)
    X_pca = X_centered @ V  # (N, n_components)
    # 白化: 除以奇异值
    X_white = X_pca / (S[:n_components].clamp(min=1e-8))
    return X_white


def compute_h0_persistence_ripser(X_np, thresh=2.0):
    """ripser Vietoris-Rips H0
    X_np: (N, d)
    返回: H0 持续图 [(b=0, d_i)] 列表
    """
    import ripser
    print(f'  [ripser] X.shape = {X_np.shape}, thresh = {thresh}')
    result = ripser.ripser(X_np, thresh=thresh)['dgms']
    h0 = result[0]  # (n, 2), cols = [birth, death]
    # 过滤 inf (最后一个连通分量一直存活)
    h0 = h0[np.isfinite(h0[:, 1])]
    print(f'  [ripser] H0 持续对: {len(h0)} 个 (过滤 inf)')
    return h0


def kde_peak_count(values, height_threshold=None, distance_threshold=None):
    """对 values 做 KDE, 数显著峰"""
    if len(values) < 10:
        return 0, None, None

    kde = gaussian_kde(values)
    xs = np.linspace(values.min(), values.max(), 1000)
    ys = kde(xs)

    # find_peaks: 峰高 ≥ height_threshold * max(ys), 间距 ≥ distance_threshold * 总宽
    height_min = height_threshold * ys.max() if height_threshold else 0.05 * ys.max()
    distance_min_samples = max(10, int(distance_threshold * len(xs))) if distance_threshold else max(10, int(0.05 * len(xs)))

    peaks, props = find_peaks(ys, height=height_min, distance=distance_min_samples)
    peak_xs = xs[peaks].tolist()
    peak_ys = ys[peaks].tolist()

    return len(peaks), peak_xs, peak_ys


def persistence_entropy(h0_pairs):
    """持续熵 E_pers = -Σ (ℓ_i / L_tot) log(ℓ_i / L_tot)
    ℓ_i = d_i - b_i, b_i = 0
    """
    lifes = h0_pairs[:, 1] - h0_pairs[:, 0]  # = d_i
    lifes = lifes[lifes > 0]
    if len(lifes) == 0:
        return 0.0
    p = lifes / lifes.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    print('=' * 70)
    print('Idea 5 TDA 现象 1: H0 持续图 + 死亡时刻 KDE 峰计数 n_peaks')
    print('=' * 70)

    # 1. 加载 embedding
    print('\n[Step 1] 加载 Stage 1 embedding')
    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float()
    N_full, d = emb.shape
    print(f'  Full embedding: {tuple(emb.shape)}')

    # 2. 子采样
    print(f'\n[Step 2] 子采样 {N_SUBSAMPLE} (seed={SEED})')
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(N_full, generator=g)[:N_SUBSAMPLE]
    emb_sub = emb[perm]
    print(f'  Sub-sampled: {tuple(emb_sub.shape)}')

    # 3. 白化
    print(f'\n[Step 3] PCA 白化到 {PCA_DIM} 维')
    emb_white = whiten_pca(emb_sub, n_components=PCA_DIM)
    print(f'  After whitening: {tuple(emb_white.shape)}, std per dim ≈ {emb_white.std(dim=0).mean().item():.3f}')

    # 4. 算 pairwise 距离 (用于后续选取阈值 + KDE 看分布)
    print('\n[Step 4] 算 pairwise 距离分布 (决定 ripser threshold)')
    # 计算 pairwise 距离的子样 (避免 5000x5000 全量)
    n_dist_sample = min(50000, N_SUBSAMPLE * (N_SUBSAMPLE - 1) // 2)
    rng = np.random.default_rng(SEED)
    i_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
    j_idx = rng.integers(0, N_SUBSAMPLE, n_dist_sample)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    X_np = emb_white.numpy()
    dists = np.linalg.norm(X_np[i_idx] - X_np[j_idx], axis=1)
    print(f'  Pairwise dist ({len(dists)} samples): min={dists.min():.3f}, '
          f'p25={np.percentile(dists, 25):.3f}, median={np.percentile(dists, 50):.3f}, '
          f'p75={np.percentile(dists, 75):.3f}, p95={np.percentile(dists, 95):.3f}, '
          f'max={dists.max():.3f}')

    # 5. ripser H0
    print('\n[Step 5] ripser Vietoris-Rips H0 持续图')
    # threshold 设为 p50 的 ~3 倍, 让合并发生在中间尺度
    thresh = float(np.percentile(dists, 90))
    print(f'  Using thresh = {thresh:.3f} (p90 of pairwise distance)')
    h0 = compute_h0_persistence_ripser(X_np, thresh=thresh)
    print(f'  H0 death statistics: min={h0[:, 1].min():.3f}, '
          f'median={np.median(h0[:, 1]):.3f}, max={h0[:, 1].max():.3f}')

    # 6. KDE 峰计数
    print('\n[Step 6] 死亡时刻 d_i KDE → 数显著峰')
    n_peaks, peak_xs, peak_ys = kde_peak_count(h0[:, 1])
    print(f'  n_peaks = {n_peaks}')
    if peak_xs:
        print(f'  Peak positions: {[round(x, 3) for x in peak_xs]}')

    # 7. 持续熵
    print('\n[Step 7] 持续熵 E_pers')
    E_pers = persistence_entropy(h0)
    print(f'  E_pers = {E_pers:.3f}')

    # 8. 判定
    print('\n[Step 8] 判定')
    if n_peaks == 1:
        verdict = 'DATA_HIERARCHY_1'
        reason = f'n_peaks=1 → 数据层级 ≈ 1, 量化算法无责 (3 层里只有 1 层有用是数据本身的特性)'
    elif n_peaks >= 2:
        # 检查尺度间距
        if len(peak_xs) >= 2:
            gaps = np.diff(sorted(peak_xs))
            min_gap = gaps.min()
            if min_gap < 0.1:
                verdict = 'DATA_HIERARCHY_1 (degenerate)'
                reason = f'n_peaks={n_peaks} 但尺度间距 {min_gap:.3f} < 0.1, 几乎不分离, 等价于 n_peaks=1'
            else:
                verdict = 'DATA_HIERARCHY_K'
                reason = f'n_peaks={n_peaks} 且尺度间距 {min_gap:.3f} ≥ 0.1 → 层级是真, 可开对准拓扑尺度方法线'
        else:
            verdict = 'INCONCLUSIVE'
            reason = 'KDE 未能稳定识别多个峰'
    else:
        verdict = 'NO_PEAKS'
        reason = 'KDE 未识别任何显著峰'

    print(f'  VERDICT: {verdict}')
    print(f'  REASON: {reason}')

    # 9. 保存
    out_json = os.path.join(OUT_DIR, 'idea5_phen1_h0_persistence.json')
    with open(out_json, 'w') as f:
        json.dump({
            'n_subsample': N_SUBSAMPLE,
            'seed': SEED,
            'pca_dim': PCA_DIM,
            'ripser_threshold': thresh,
            'pairwise_dist_percentiles': {
                'min': float(dists.min()),
                'p25': float(np.percentile(dists, 25)),
                'p50': float(np.percentile(dists, 50)),
                'p75': float(np.percentile(dists, 75)),
                'p95': float(np.percentile(dists, 95)),
                'max': float(dists.max()),
            },
            'h0_stats': {
                'n_pairs': len(h0),
                'death_min': float(h0[:, 1].min()),
                'death_median': float(np.median(h0[:, 1])),
                'death_max': float(h0[:, 1].max()),
                'death_mean': float(h0[:, 1].mean()),
                'death_std': float(h0[:, 1].std()),
            },
            'kde_peaks': {
                'n_peaks': n_peaks,
                'peak_positions': peak_xs,
            },
            'E_pers': E_pers,
            'verdict': {'status': verdict, 'reason': reason},
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task10_persistent_homology_tda.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # 10. KDE plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        kde = gaussian_kde(h0[:, 1])
        xs = np.linspace(h0[:, 1].min(), h0[:, 1].max(), 1000)
        ys = kde(xs)
        ax.plot(xs, ys, 'b-', lw=2, label=f'KDE (n_peaks={n_peaks})')
        if peak_xs:
            for px, py in zip(peak_xs, peak_ys):
                ax.axvline(px, color='r', ls='--', alpha=0.5)
                ax.annotate(f'{px:.3f}', (px, py), xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Death scale d_i')
        ax.set_ylabel('KDE density')
        ax.set_title(f'Idea 5 TDA: H0 death-time KDE\nn_peaks = {n_peaks}, E_pers = {E_pers:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_png = os.path.join(OUT_DIR, 'h0_persistence_kde.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] 画图失败: {e}')

    return n_peaks, E_pers, verdict


if __name__ == '__main__':
    main()
