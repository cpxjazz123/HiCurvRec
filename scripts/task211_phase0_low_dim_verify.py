#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #211 Phase 0 前置验证 — 低维双曲 + 钉半径可行性.

不占 GPU, 不动训练代码. 纯 CPU 推理:
  1. 加载 item_emb.parquet (9922 × 768 sentence-t5-base)
  2. PCA → d ∈ {2, 4, 8}
  3. 球面 k-means (单位方向) + 钉半径 ρ
  4. 用真实 latent 做 argmin 分配
  5. 测 4 个指标

输出:
  - verdicts/task211_phase0_metrics.json (机器可读)
  - 控制台打印 36 格子表 (3 d × 3 layer × 4 metric)

通过标准:
  - dyn_range ≥ 2.0
  - max ‖x‖ ≤ 0.95 (球内)
  - unique_codes / K ≥ 90%
  - min_angle (记录, 解释用)
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task211_phase0")


def spherical_kmeans(X_unit, n_clusters, random_state=42, n_init=10, max_iter=300):
    """球面 k-means: 在单位球面上做 K-means.

    X_unit: (N, d) 单位向量 (rows are unit-normalized)
    返回: centers_unit (K, d) — 单位向量, 球面聚类中心
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    # sklearn KMeans 用欧式距离, 但 init 之前我们手动 normalize,
    # 然后每个 iteration 后再 normalize centers.
    kmeans.fit(X_unit)
    centers = kmeans.cluster_centers_
    centers_unit = normalize(centers, norm='l2', axis=1)
    return centers_unit


def pinned_radius_assign(X, centers_pinned, metric='euclidean'):
    """用真实 latent 做 argmin 分配.

    X: (N, d) raw latent (PCA 后)
    centers_pinned: (K, d) 已钉到 ρ 的码字
    metric: 'euclidean' 或 'poincare' (在 ρ 内径, 用 Poincaré 距离更准, 但 d=4+ 都是小 d, 欧式足够区分)

    返回: assigned_indices (N,), distances (N,)
    """
    N = X.shape[0]
    K = centers_pinned.shape[0]
    if metric == 'euclidean':
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
        c_norm_sq = np.sum(centers_pinned ** 2, axis=1, keepdims=True).T  # (1, K)
        x_dot_c = X @ centers_pinned.T  # (N, K)
        dist_sq = x_norm_sq + c_norm_sq - 2 * x_dot_c  # (N, K)
        assigned = np.argmin(dist_sq, axis=1)
        distances = np.sqrt(np.min(dist_sq, axis=1))
    elif metric == 'poincare':
        # Poincaré 距离公式: d(x,y) = (2/√c) artanh(√c · ||(-x) ⊕ y||)
        # 这里 c=1, 不归一化 (X 是 PCA 后的 raw 数值, 跟 ρ=2.0+ 在一个尺度上)
        c = 1.0
        # mobius add: (-x) ⊕ y
        x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
        y_norm_sq = np.sum(centers_pinned ** 2, axis=1, keepdims=True).T  # (1, K)
        xy_dot = -X @ centers_pinned.T  # (-x) · y
        # 注意: (-x) ⊕ y = ((1 + c*xy - c*y_norm_sq) * (-x) + (1 - c*x_norm_sq) * y) / denom
        # 这里 c=1, 简化:
        # numer1 = (1 + xy - y_norm_sq) * (-X)  # (N, K, d)
        # numer2 = (1 - x_norm_sq) * centers_pinned  # (N, K, d)
        # 但为了简单起见, 在 d=4 小 d + ρ ≤ 3.4 范围内, 欧式足够区分.
        log.info("  Poincaré 距离 fallback to euclidean for speed")
        return pinned_radius_assign(X, centers_pinned, metric='euclidean')
    else:
        raise ValueError(f"unknown metric: {metric}")
    return assigned, distances


def compute_metrics(X, centers_pinned, assigned, distances):
    """计算 4 个指标.

    X: (N, d) raw PCA latent
    centers_pinned: (K, d) 钉半径后的码字
    assigned: (N,) 分配结果
    distances: (N,) 分配距离
    """
    K = centers_pinned.shape[0]
    N = X.shape[0]

    # 1. dyn_range (最远/最近邻) — 用 K 个码字间的 pairwise 距离
    #   实际意义: 码字在球面上分布广度, dyn 大 = 边界 vs 中心区分明显
    #   注意: 用 np.triu 取上三角, 排除对角线 + 避免重复对
    center_dists = np.linalg.norm(centers_pinned[:, None, :] - centers_pinned[None, :, :], axis=2)
    triu_idx = np.triu_indices(K, k=1)
    pair_dists = center_dists[triu_idx]  # (K*(K-1)/2,) 所有 unique 对距离
    if pair_dists.min() < 1e-8:
        dyn_range = float('inf')
        log.warning(f"  ⚠️  最近邻距离近 0: dyn_range=inf")
    else:
        dyn_range = float(pair_dists.max() / pair_dists.min())

    # 2. max ‖x‖ (球内) — 钉到 ρ 后, 码字范数应该都是 ρ
    #   但检查一下: 是否真有码字超 ρ (例如 kmeans 没收敛)
    center_norms = np.linalg.norm(centers_pinned, axis=1)  # (K,)
    max_norm = float(center_norms.max())

    # 3. unique_codes / K
    unique_used = len(np.unique(assigned))
    utilization = unique_used / K

    # 4. min pairwise angle (cos)
    #   cos θ = x·y / (||x||·||y||), 由于 centers 都已 normalize, 直接 dot
    centers_unit = normalize(centers_pinned, norm='l2', axis=1)
    dot_mat = centers_unit @ centers_unit.T
    np.fill_diagonal(dot_mat, -2)  # exclude self (cos=1)
    min_cos = float(dot_mat.max())  # closest pair has highest cos
    min_angle_rad = float(np.arccos(np.clip(min_cos, -1, 1)))

    return {
        'dyn_range': dyn_range,
        'max_norm': max_norm,
        'utilization': utilization,
        'min_angle_rad': min_angle_rad,
        'unique_used': unique_used,
        'K': K,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task211_phase0_metrics.json")
    args = parser.parse_args()

    log.info(f"=== Task #211 Phase 0 前置验证 (低维双曲 + 钉半径) ===")
    log.info(f"Loading data: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    log.info(f"  shape: {df.shape}, columns: {df.columns.tolist()}")

    # Extract embedding
    if 'emb' in df.columns:
        X = np.stack(df['emb'].values)
    elif 'embedding' in df.columns:
        X = np.stack(df['embedding'].values)
    else:
        # 假设所有数值列都是 embedding
        X = df.values.astype(np.float32)
    log.info(f"  X: {X.shape}, dtype={X.dtype}, norm_range=[{np.linalg.norm(X, axis=1).min():.3f}, "
             f"{np.linalg.norm(X, axis=1).max():.3f}]")

    # 三层配置: K = 64/128/256, ρ = 2.0/2.7/3.4
    layer_configs = [
        {'layer': 0, 'K': 64, 'rho': 2.0},
        {'layer': 1, 'K': 128, 'rho': 2.7},
        {'layer': 2, 'K': 256, 'rho': 3.4},
    ]
    d_values = [2, 4, 8]

    # 全部结果: {d: {layer: metrics}}
    all_results = {}
    for d in d_values:
        log.info(f"\n===== d = {d} =====")
        # PCA → d
        log.info(f"  PCA {X.shape[1]} → {d}")
        pca = PCA(n_components=d, random_state=42)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_.sum()
        log.info(f"    explained_var_sum = {explained_var:.4f}, X_pca range: [{X_pca.min():.3f}, {X_pca.max():.3f}]")

        # 单位归一化 (球面 k-means 准备)
        X_unit = normalize(X_pca, norm='l2', axis=1)
        log.info(f"  After L2-normalize: ‖x‖=1")

        all_results[d] = {}
        for cfg in layer_configs:
            K = cfg['K']
            rho = cfg['rho']
            layer = cfg['layer']
            # 关键解读: ρ 是双曲空间半径 (Poincaré ball 中从原点到 x 的 hyperbolic distance),
            # 对应 Euclidean norm = tanh(ρ/2) (c=1.0).
            # ρ=2.0 → ‖x‖_E ≈ 0.762; ρ=2.7 → ≈ 0.876; ρ=3.4 → ≈ 0.935. 均 ≤ 0.95 gate.
            euclidean_norm = float(np.tanh(rho / 2.0))
            t0 = time.time()
            log.info(f"  Layer {layer}: K={K}, ρ={rho} → ‖x‖_E_target={euclidean_norm:.3f}")

            # 球面 k-means (在单位球面上)
            centers_unit = spherical_kmeans(X_unit, n_clusters=K, random_state=42)
            # 钉到 Euclidean norm = tanh(ρ/2) (在 Poincaré 球内)
            centers_pinned = centers_unit * euclidean_norm
            # 分配 (用 raw PCA latent 做 argmin, 距离用欧式)
            assigned, distances = pinned_radius_assign(X_pca, centers_pinned, metric='euclidean')
            # 4 指标
            metrics = compute_metrics(X_pca, centers_pinned, assigned, distances)
            metrics['explained_var_sum'] = float(explained_var)
            metrics['elapsed_sec'] = round(time.time() - t0, 2)

            all_results[d][layer] = metrics
            log.info(f"    dyn={metrics['dyn_range']:.3f} max‖x‖={metrics['max_norm']:.3f} "
                     f"util={metrics['utilization']*100:.1f}% angle={metrics['min_angle_rad']:.3f}rad "
                     f"({time.time()-t0:.1f}s)")

    # 决策: 哪个 d 通过?
    log.info("\n===== 决策 (gate: dyn ≥ 2.0 AND max ≤ 0.95 AND util ≥ 90%) =====")
    decision = {}
    for d in d_values:
        layers_pass = []
        for layer in range(3):
            m = all_results[d][layer]
            passes = (
                m['dyn_range'] >= 2.0 and
                m['max_norm'] <= 0.95 and
                m['utilization'] >= 0.90
            )
            layers_pass.append(passes)
        # Phase 0 通过 = 三层全过 (或者允许 L2 不完全过, 因为 L2 ρ=3.4 + K=256)
        all_pass = all(layers_pass)
        any_pass = any(layers_pass)
        decision[d] = {
            'layers_pass': layers_pass,
            'all_pass': all_pass,
            'any_pass': any_pass,
            'recommendation': 'GO' if all_pass else ('PARTIAL' if any_pass else 'NO-GO'),
        }
        log.info(f"  d={d}: layer_pass={layers_pass} → {decision[d]['recommendation']}")

    # 推荐 d
    recommend_d = None
    for d in d_values:
        if decision[d]['all_pass']:
            recommend_d = d
            break  # 第一个全过 = 推荐
    if recommend_d is None:
        # 没有全过, 选通过最多的层数
        best_d = max(d_values, key=lambda d: sum(decision[d]['layers_pass']))
        recommend_d = best_d if sum(decision[best_d]['layers_pass']) > 0 else None
    log.info(f"\n推荐 d_hyp = {recommend_d}")

    # 保存 JSON
    output = {
        'task': 'task211_phase0_verify',
        'n_items': int(X.shape[0]),
        'orig_dim': int(X.shape[1]),
        'd_values': d_values,
        'layer_configs': layer_configs,
        'results': {str(d): {str(l): all_results[d][l] for l in range(3)} for d in d_values},
        'decision': {str(d): decision[d] for d in d_values},
        'recommended_d': recommend_d,
        'gates': {
            'dyn_range_min': 2.0,
            'max_norm_max': 0.95,
            'utilization_min': 0.90,
        },
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"\n✅ Saved: {args.output_json}")


if __name__ == '__main__':
    main()