#!/usr/bin/env python3
"""Task #37 — 几何区分性诊断.

承接 Task #36: 3 个 subspace (sphere/euclid/hyperbolic) + fused 在 MCKG 都 gap ≥ 0.016.
本任务评估:
  A. 同一 item 在 3 subspace 的 top-K 最近邻 overlap (Jaccard)
     → 高 overlap: 3 子空间冗余, PM-RQ 动机弱
     → 低 overlap: 3 子空间几何区分, PM-RQ 动机强

  B. subspace 间逐 item cosine 分布
     → 期望: 同 subspace 自身 cosine 高 (聚集), 跨 subspace cosine 低 (区分)

  C. fused_item 的最近邻 vs (subspace_0 ∪ subspace_1 ∪ subspace_2) 各自最近邻的差异
     → 看融合是否引入了新信息

输出: products/task37_subspace_geometry/task37_summary.json
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUTPUT_DIR = ROOT / 'products/task37_subspace_geometry'
OUTPUT_DIR.mkdir(exist_ok=True)

K = 20  # top-K 用于 overlap 评估


def l2_norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def topk_indices(emb_norm, k):
    """emb_norm: (N, D), L2-normalized. 返回 (N, k) top-K 最近邻 idx."""
    sim = emb_norm @ emb_norm.T  # (N, N)
    np.fill_diagonal(sim, -2.0)
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    # 排序 (按 sim 降序)
    row_idx = np.arange(sim.shape[0])[:, None]
    top_sim = sim[row_idx, top]
    order = np.argsort(-top_sim, axis=1)
    return top[row_idx, order]


def jaccard(a, b):
    """两个 top-K 集合的 Jaccard = |a ∩ b| / |a ∪ b|."""
    a_set = set(a.tolist())
    b_set = set(b.tolist())
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


def main():
    print('========== Task #37 subspace 几何区分性诊断 ==========')
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    subspace_item = mckg['subspace_item'].numpy()
    fused_item = mckg['fused_item'].numpy()
    kappas = mckg['kappas']
    N = subspace_item.shape[1]
    print(f'kappas = {kappas}, N_items = {N}, K = {K}')

    spaces = {
        'subspace_0_sphere(κ+5.05)': l2_norm(subspace_item[0]),
        'subspace_1_euclid(κ-0.08)': l2_norm(subspace_item[1]),
        'subspace_2_hyperbolic(κ-5.04)': l2_norm(subspace_item[2]),
        'fused(64d)': l2_norm(fused_item),
    }

    top_knn = {}
    print('\n[A] 计算 4 个空间的 top-K 最近邻...')
    for name, emb in spaces.items():
        top_knn[name] = topk_indices(emb, K)
        print(f'  {name}: done (shape={top_knn[name].shape})')

    # ---- B. subspace 间 Jaccard overlap ----
    print('\n[B] 子空间间 top-K Jaccard overlap...')
    pairs = [
        ('subspace_0_sphere(κ+5.05)', 'subspace_1_euclid(κ-0.08)'),
        ('subspace_0_sphere(κ+5.05)', 'subspace_2_hyperbolic(κ-5.04)'),
        ('subspace_1_euclid(κ-0.08)', 'subspace_2_hyperbolic(κ-5.04)'),
        ('subspace_0_sphere(κ+5.05)', 'fused(64d)'),
        ('subspace_1_euclid(κ-0.08)', 'fused(64d)'),
        ('subspace_2_hyperbolic(κ-5.04)', 'fused(64d)'),
    ]
    jaccard_results = {}
    for a, b in pairs:
        jac_per_item = []
        for i in range(0, N, 200):
            chunk = min(200, N - i)
            for j in range(chunk):
                jac_per_item.append(jaccard(top_knn[a][i + j], top_knn[b][i + j]))
        jac_mean = float(np.mean(jac_per_item))
        jac_std = float(np.std(jac_per_item))
        jaccard_results[f'{a} vs {b}'] = {
            'mean_jaccard': jac_mean,
            'std_jaccard': jac_std,
            'median_jaccard': float(np.median(jac_per_item)),
            'p25_jaccard': float(np.percentile(jac_per_item, 25)),
            'p75_jaccard': float(np.percentile(jac_per_item, 75)),
        }
        print(f'  {a} vs {b}: mean={jac_mean:.4f}, std={jac_std:.4f}, '
              f'median={jaccard_results[f"{a} vs {b}"]["median_jaccard"]:.4f}')

    # ---- C. 跨 subspace cosine 分布 ----
    print('\n[C] 跨 subspace 逐 item cosine 分布...')
    cross_cos = {}
    for a, b in pairs:
        emb_a = spaces[a]
        emb_b = spaces[b]
        # 随机采样 1000 个 item 的 cos sim 分布
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(N, size=min(1000, N), replace=False)
        cos_vals = (emb_a[idx_sample] * emb_b[idx_sample]).sum(axis=-1)
        cross_cos[f'{a} vs {b}'] = {
            'mean': float(cos_vals.mean()),
            'std': float(cos_vals.std()),
            'median': float(np.median(cos_vals)),
            'min': float(cos_vals.min()),
            'max': float(cos_vals.max()),
        }
        print(f'  {a} vs {b}: mean_cos={cos_vals.mean():.4f}, '
              f'std={cos_vals.std():.4f}, median={np.median(cos_vals):.4f}')

    # ---- D. 几何区分性判据 ----
    print('\n[D] 几何区分性判据...')
    s0_vs_s1 = jaccard_results['subspace_0_sphere(κ+5.05) vs subspace_1_euclid(κ-0.08)']['mean_jaccard']
    s0_vs_s2 = jaccard_results['subspace_0_sphere(κ+5.05) vs subspace_2_hyperbolic(κ-5.04)']['mean_jaccard']
    s1_vs_s2 = jaccard_results['subspace_1_euclid(κ-0.08) vs subspace_2_hyperbolic(κ-5.04)']['mean_jaccard']
    avg_cross_subspace_jaccard = (s0_vs_s1 + s0_vs_s2 + s1_vs_s2) / 3

    print(f'  3 subspace 平均 cross-Jaccard = {avg_cross_subspace_jaccard:.4f}')
    if avg_cross_subspace_jaccard < 0.1:
        verdict = 'HIGH_DISTINCT (Jaccard<0.1): 3 子空间高度区分, PM-RQ 几何优势强'
    elif avg_cross_subspace_jaccard < 0.3:
        verdict = 'MEDIUM_DISTINCT (0.1≤Jaccard<0.3): 子空间部分区分, PM-RQ 仍可能有效'
    else:
        verdict = 'LOW_DISTINCT (Jaccard≥0.3): 3 子空间几何冗余, PM-RQ 动机被削弱'

    print(f'  判据: {verdict}')

    summary = {
        'task': 'Task #37 subspace 几何区分性诊断',
        'mckg_geometry': {
            'kappas': [float(x) for x in kappas],
            'N_items': N,
            'K': K,
        },
        'jaccard_results': jaccard_results,
        'cross_cosine_distribution': cross_cos,
        'avg_cross_subspace_jaccard': avg_cross_subspace_jaccard,
        'verdict': verdict,
    }

    out_json = OUTPUT_DIR / 'task37_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out_json}')

    # ---- Final summary table ----
    print('\n========== Task #37 Summary ==========')
    print(f'Top-{K} Jaccard overlap (子空间越低越好):')
    for k, v in jaccard_results.items():
        print(f'  {k}: {v["mean_jaccard"]:.4f}')
    print(f'\n判据: {verdict}')


if __name__ == '__main__':
    main()