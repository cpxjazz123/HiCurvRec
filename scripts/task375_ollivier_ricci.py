#!/usr/bin/env python3
"""Task 375: Ollivier-Ricci 曲率分析 — 转移图

判断转移图整体曲率符号：
  κ(u,v) > 0 → 正曲率 → 图偏向"球面/星状" → 深层宜用球面几何
  κ(u,v) < 0 → 负曲率 → 图偏向"树状/双曲" → 深层宜用双曲几何
  κ(u,v) ≈ 0 → 平坦 → 深层用欧氏几何即可

Ollivier-Ricci 公式（离散图近似）：
  κ(u,v) ≈ 1 - (W1(π_u, π_v)) / d(u,v)
  其中 π_u = 邻居节点的均匀分布（1/deg(u) 概率）
        W1 = Earth Mover Distance（均匀分布间的 Wasserstein-1 距离）

对无权无向图，W1(π_u, π_v) = sum_{w} |π_u(w) - π_v(w)| / 2 （总变差距离的闭式近似）
  = sum_{w} |1/deg(u) - 1/deg(v)| / 2，对共同邻居 w

参考文献: Ollivier, R. (2009). Ricci curvature of Markov chains on metric spaces.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
import argparse

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
from task370_network_structure import build_transition_graph, build_adj_list

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task375_ollivier_ricci'
os.makedirs(OUT_DIR, exist_ok=True)


def build_transition_graph_from_edges(n_items=11924, n_users=3000, min_count=2):
    """复用 task370 的转移图构建逻辑（重建一次）。"""
    transitions = defaultdict(lambda: defaultdict(int))
    user_count = 0
    import tensorflow as tf

    train_files = sorted(os.listdir(f'{GRID}/data/amazon_data/toys/training/'))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(f'{GRID}/data/amazon_data/toys/training/', fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            seq_data = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq_data) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            for pos in range(len(seq_data) - 1):
                a, b = seq_data[pos], seq_data[pos + 1]
                if a < n_items and b < n_items:
                    transitions[a][b] += 1
        if user_count > n_users:
            break

    edges = []
    for a, b2 in transitions.items():
        for b, cnt in b2.items():
            if cnt >= min_count:
                edges.append((a, b, cnt))
    return edges, dict(transitions)


def ollivier_ricci_edge_curvature(u, v, adj, neighbors):
    """对边 (u,v) 计算 Ollivier-Ricci 曲率近似。

    均匀概率分布：π_u(w) = 1/deg(u)，对 w ∈ N(u)
    W1(π_u, π_v) = (1/2) * sum_w |π_u(w) - π_v(w)| （总变差距离）

    近似公式：κ(u,v) ≈ 1 - W1(π_u, π_v) / d(u,v)
    若 u,v 直接相连：d(u,v)=1
    """
    deg_u = len(neighbors.get(u, []))
    deg_v = len(neighbors.get(v, []))
    if deg_u == 0 or deg_v == 0:
        return 0.0

    nb_u = set(neighbors.get(u, []))
    nb_v = set(neighbors.get(v, []))

    # W1 between uniform distributions on neighbors (TV distance)
    # W1 = (1/2) * sum_w |1/deg_u - 1/deg_v| over all w in nb_u ∪ nb_v
    all_nb = nb_u | nb_v
    w1 = 0.0
    for w in all_nb:
        pu = 1.0 / deg_u if w in nb_u else 0.0
        pv = 1.0 / deg_v if w in nb_v else 0.0
        w1 += abs(pu - pv)
    w1 *= 0.5  # W1 = (1/2) * sum |π_u - π_v|

    d_uv = 1.0  # directly connected
    kappa = 1.0 - w1 / d_uv
    return kappa


def all_pairs_shortest_path(adj, nodes):
    """对所有节点对计算最短路径距离（BFS from each node）。"""
    from collections import deque
    n = len(nodes)
    node_list = list(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    INF = 999

    dist = np.full((n, n), INF, dtype=np.float32)
    for i, node in enumerate(node_list):
        dist[i, i] = 0.0
        visited = {node: 0}
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            for nxt in adj.get(cur, []):
                if nxt not in visited:
                    visited[nxt] = visited[cur] + 1
                    queue.append(nxt)
        for j, nbr in enumerate(node_list):
            if nbr in visited:
                dist[i, j] = visited[nbr]
    return dist, node_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--min-count', type=int, default=2)
    args = parser.parse_args()

    print('=' * 70)
    print('Task 375: Ollivier-Ricci 曲率分析 — 转移图')
    print('=' * 70)

    # Build transition graph (same as task370)
    print('\n[Building transition graph]')
    edges, transition_counts = build_transition_graph_from_edges(
        n_items=11924, n_users=args.n_users, min_count=args.min_count)
    adj = defaultdict(set)
    for a, b, cnt in edges:
        adj[a].add(b)
        adj[b].add(a)  # undirected for curvature
    adj = dict(adj)

    all_nodes = set(adj.keys())
    print(f'  Edges: {len(edges)}, Nodes: {len(all_nodes)}')

    # Compute average degree
    degrees = [len(adj[n]) for n in all_nodes]
    print(f'  Avg degree: {np.mean(degrees):.2f}, Median: {np.median(degrees):.1f}')

    # Compute all-pairs shortest path for d(u,v)
    print('\n[Computing shortest path distances]')
    dist_mat, node_list = all_pairs_shortest_path(adj, all_nodes)
    n = len(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # Compute Ollivier-Ricci for each edge
    print(f'\n[Computing Ollivier-Ricci curvature for {len(edges)} edges]')
    kappas = []
    skipped = 0
    for a, b, cnt in edges:
        # For undirected graph, compute curvature for (a,b) if both have neighbors
        if len(adj.get(a, [])) == 0 or len(adj.get(b, [])) == 0:
            skipped += 1
            continue
        kappa = ollivier_ricci_edge_curvature(a, b, adj, adj)
        kappas.append(kappa)

    print(f'  Computed: {len(kappas)} edges, Skipped (isolated): {skipped}')
    kappas = np.array(kappas)

    # Statistics
    mean_kappa = float(np.mean(kappas))
    median_kappa = float(np.median(kappas))
    std_kappa = float(np.std(kappas))
    pos_frac = float(np.mean(kappas > 0))
    neg_frac = float(np.mean(kappas < 0))

    print(f'\n=== Ollivier-Ricci Curvature Summary ===')
    print(f'  Mean:   {mean_kappa:+.6f}')
    print(f'  Median: {median_kappa:+.6f}')
    print(f'  Std:    {std_kappa:.6f}')
    print(f'  Positive (>0): {pos_frac*100:.1f}% of edges')
    print(f'  Negative (<0): {neg_frac*100:.1f}% of edges')
    print(f'  Zero (=0): {(1-pos_frac-neg_frac)*100:.1f}% of edges')
    print(f'  Min: {kappas.min():+.6f}, Max: {kappas.max():+.6f}')

    # Histogram
    bins = np.linspace(kappas.min() - 0.01, kappas.max() + 0.01, 20)
    hist, bin_edges = np.histogram(kappas, bins=bins)
    print('\nHistogram:')
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        bar = '#' * int(hist[i] / max(hist.max(), 1) * 40)
        print(f'  [{lo:+.4f}, {hi:+.4f}): {int(hist[i]):3d} {bar}')

    # Save results
    results = {
        'task': 'task375_ollivier_ricci',
        'method': 'Ollivier-Ricci curvature (uniform neighbor distribution, TV distance)',
        'n_edges': len(edges),
        'n_nodes': len(all_nodes),
        'n_computed': len(kappas),
        'n_skipped': skipped,
        'avg_degree': float(np.mean(degrees)),
        'curvature': {
            'mean': mean_kappa,
            'median': median_kappa,
            'std': std_kappa,
            'min': float(kappas.min()),
            'max': float(kappas.max()),
            'fraction_positive': pos_frac,
            'fraction_negative': neg_frac,
        },
        'per_edge': [{'u': int(a), 'v': int(b), 'kappa': float(k)} for (a, b, _), k in zip(edges, kappas)],
    }
    json_path = f'{OUT_DIR}/ollivier_ricci.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n✅ Saved: {json_path}')

    # Verdict
    write_verdict(results)
    print(f'\n✅ Saved: {OUT_DIR}/verdict.md')


def write_verdict(results):
    path = f'{OUT_DIR}/verdict.md'
    m = results['curvature']
    mean_k = m['mean']
    pos_frac = m['fraction_positive']
    neg_frac = m['fraction_negative']

    # Interpretation
    if mean_k < -0.05:
        interpretation = '负曲率（树状/双曲倾向）'
        recommendation = '深层宜用双曲几何（H-E-E-E）'
        narrative = (
            '转移图呈负曲率，说明购买顺序关系具有"树状/星形"拓扑特征。'
            '负曲率意味着图中任意三点有唯一一个"中点"附近的汇聚结构，'
            '这正是双曲空间擅长表达的几何特征。'
            '因此深层用双曲几何（H-E-E-E 或 H-H-E-E）可能有价值。'
        )
    elif mean_k > 0.05:
        interpretation = '正曲率（球面/团状倾向）'
        recommendation = '深层宜用球面几何（需球面RNN/SNN，不在当前计划内）'
        narrative = (
            '转移图呈正曲率，说明购买顺序关系偏向"局部团簇"结构。'
            '正曲率意味着局部邻居趋于聚集（类似球面几何），'
            '但这对推荐系统的深层结构建模价值有限（推荐不需要全局球面几何）。'
        )
    else:
        interpretation = '近零曲率（平坦/欧氏倾向）'
        recommendation = '深层用欧氏几何即可（H-E-E-E 的 E-E-E 部分已足够）'
        narrative = (
            '转移图的 Ollivier-Ricci 曲率接近 0，说明其结构接近"平坦图"。'
            '在平坦图上，双曲几何和球面几何均无特别优势，深层用标准欧氏残差即可。'
            '这与 task358 结论一致：深层 residual 的主要问题是"信息已被 L1 提取"，不是"几何选错"。'
        )

    lines = [
        '# Task 375 Verdict: Ollivier-Ricci 曲率分析 — 转移图',
        '',
        '## 设计',
        '',
        f'- **图**: {results["n_edges"]} 边, {results["n_nodes"]} 节点, avg_degree={results["avg_degree"]:.2f}',
        f'- **方法**: Ollivier-Ricci 曲率（均匀邻居分布 + 总变差距离）',
        f'- **公式**: κ(u,v) ≈ 1 - W₁(π_u, π_v) / d(u,v), d(u,v)=1（直接相连）',
        '',
        '## 曲率统计',
        '',
        f'| 指标 | 值 |',
        f'|-------|-----|',
        f'| Mean κ | {m["mean"]:+.6f} |',
        f'| Median κ | {m["median"]:+.6f} |',
        f'| Std κ | {m["std"]:.6f} |',
        f'| κ > 0 (正曲率) | {pos_frac*100:.1f}% |',
        f'| κ < 0 (负曲率) | {neg_frac*100:.1f}% |',
        f'| κ ≈ 0 (平坦) | {(1-pos_frac-neg_frac)*100:.1f}% |',
        f'| Min κ | {m["min"]:+.6f} |',
        f'| Max κ | {m["max"]:+.6f} |',
        '',
        '## 解读',
        '',
        f'**整体曲率**: {interpretation}',
        f'**推荐几何**: {recommendation}',
        '',
        f'**直觉解释**: {narrative}',
        '',
        '## 对 H-E-E-E 实验的影响',
        '',
    ]

    if mean_k < -0.05:
        lines.extend([
            f'→ 转移图负曲率（κ_mean={mean_k:+.4f}）支持 **H-E-E-E** 深层用双曲几何',
            f'→ 但当前 RQ-VAE 深层用欧氏几何，转移图结构可能已被欧氏残差以次优方式近似',
            f'→ 如果 H-E-E-E 能更好地保持转移图结构 → R@10 提升会是直接证据',
            '→ 注意：转移图仅 213 边/370 节点，曲率估计方差可能较大（需 bootstrap CI）',
        ])
    elif mean_k > 0.05:
        lines.extend([
            f'→ 转移图正曲率（κ_mean={mean_k:+.4f}），双曲几何可能不适用',
            '→ 但正曲率对推荐系统的价值有限（局部团簇不等于顺序规律）',
            '→ 更值得关注的是 Copurchase 图（更密集）的曲率',
        ])
    else:
        lines.extend([
            f'→ 转移图近零曲率（κ_mean={mean_k:+.4f}），欧氏几何已足够',
            '→ 深层双曲几何对转移图的额外价值有限',
            '→ 建议优先验证转移图结构是否在 H-E-E-E 中被更好地转化为 R@10',
        ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
