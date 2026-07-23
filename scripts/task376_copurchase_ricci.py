#!/usr/bin/env python3
"""Task 376: Copurchase 图 Ollivier-Ricci 曲率分析

与 task375 相同的方法，但针对 Copurchase 图（57K 边，8132 节点，更密集）。
如果 Copurchase 呈负曲率 → 支持深层双曲几何；如果近零 → 支持欧氏几何。
"""

import os, sys, json, argparse
from collections import defaultdict
import numpy as np
import tensorflow as tf

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task376_copurchase_ricci'
os.makedirs(OUT_DIR, exist_ok=True)


def build_copurchase_graph(n_items=11924, n_users=3000, window=5, min_weight=2):
    """同 task370 的共购图构建逻辑。"""
    cooccur = defaultdict(lambda: defaultdict(int))
    user_count = 0
    train_files = sorted(os.listdir(f'{GRID}/data/amazon_data/toys/training/'))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(f'{GRID}/data/amazon_data/toys/training/', fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            for pos_i in range(len(seq)):
                item_i = seq[pos_i]
                if item_i >= n_items:
                    continue
                for pos_j in range(max(0, pos_i - window), min(len(seq), pos_i + window + 1)):
                    if pos_i == pos_j:
                        continue
                    item_j = seq[pos_j]
                    if item_j >= n_items:
                        continue
                    if item_i < item_j:
                        cooccur[item_i][item_j] += 1
                    else:
                        cooccur[item_j][item_i] += 1
        if user_count > n_users:
            break

    edges = []
    cooccur_full = {}
    for i, j2 in cooccur.items():
        for j, w in j2.items():
            if i < n_items and j < n_items:
                cooccur_full[(i, j)] = w
                cooccur_full[(j, i)] = w
                if w >= min_weight:
                    edges.append((i, j, w))
    print(f'Copurchase: {len(edges)} edges, {user_count} users')
    return edges, cooccur_full


def ollivier_ricci_edge(u, v, adj):
    """均匀邻居分布间的总变差距离 → Ollivier-Ricci 曲率近似。"""
    nb_u = adj.get(u, set())
    nb_v = adj.get(v, set())
    deg_u = len(nb_u)
    deg_v = len(nb_v)
    if deg_u == 0 or deg_v == 0:
        return 0.0
    pu = 1.0 / deg_u
    pv = 1.0 / deg_v
    # W1 = (1/2) * sum_w |π_u(w) - π_v(w)|
    all_nb = nb_u | nb_v
    w1 = 0.0
    for w in all_nb:
        puw = pu if w in nb_u else 0.0
        pvw = pv if w in nb_v else 0.0
        w1 += abs(puw - pvw)
    w1 *= 0.5
    return 1.0 - w1  # κ(u,v) ≈ 1 - W1/d(u,v), d=1 for direct edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min-weight', type=int, default=2)
    args = parser.parse_args()
    print('=' * 70)
    print('Task 376: Copurchase 图 Ollivier-Ricci 曲率分析')
    print('=' * 70)

    print('\n[Building Copurchase graph]')
    edges, cooccur_full = build_copurchase_graph(
        n_items=11924, n_users=args.n_users, window=args.window, min_weight=args.min_weight)

    # Build undirected adjacency
    adj = defaultdict(set)
    for i, j, w in edges:
        adj[i].add(j)
        adj[j].add(i)
    adj = dict(adj)
    nodes = list(adj.keys())
    print(f'Nodes with edges: {len(nodes)}, Edges: {len(edges)}')

    degrees = [len(adj.get(n, [])) for n in nodes]
    print(f'Avg degree: {np.mean(degrees):.2f}, Median: {np.median(degrees):.1f}')

    print(f'\n[Computing Ollivier-Ricci for {len(edges)} edges]')
    kappas = []
    for idx, (u, v, w) in enumerate(edges):
        if idx % 10000 == 0:
            print(f'  {idx}/{len(edges)} edges processed')
        k = ollivier_ricci_edge(u, v, adj)
        kappas.append(k)

    kappas = np.array(kappas)
    m = {
        'mean': float(np.mean(kappas)),
        'median': float(np.median(kappas)),
        'std': float(np.std(kappas)),
        'min': float(kappas.min()),
        'max': float(kappas.max()),
        'frac_pos': float(np.mean(kappas > 0)),
        'frac_neg': float(np.mean(kappas < 0)),
        'frac_zero': float(np.mean(np.abs(kappas) < 1e-8)),
    }

    print(f'\n=== Copurchase Ollivier-Ricci Summary ===')
    print(f'Mean:   {m["mean"]:+.6f}')
    print(f'Median: {m["median"]:+.6f}')
    print(f'Std:    {m["std"]:.6f}')
    print(f'κ > 0:  {m["frac_pos"]*100:.1f}%')
    print(f'κ < 0:  {m["frac_neg"]*100:.1f}%')
    print(f'κ ≈ 0:  {m["frac_zero"]*100:.1f}%')
    print(f'Range:   [{m["min"]:+.4f}, {m["max"]:+.4f}]')

    # Histogram
    bins = np.linspace(kappas.min(), kappas.max(), 20)
    hist, bedges = np.histogram(kappas, bins=bins)
    print('\nHistogram:')
    for i, (lo, hi) in enumerate(zip(bedges[:-1], bedges[1:])):
        bar = '#' * int(hist[i] / max(hist.max(), 1) * 50)
        print(f'  [{lo:+.4f}, {hi:+.4f}): {int(hist[i]):5d} {bar}')

    results = {
        'task': 'task376_copurchase_ollivier_ricci',
        'method': 'Ollivier-Ricci (uniform neighbor, TV distance)',
        'n_edges': len(edges),
        'n_nodes': len(nodes),
        'avg_degree': float(np.mean(degrees)),
        'curvature': m,
        'per_edge': [{'u': int(u), 'v': int(v), 'kappa': float(k)} for (u, v, _), k in zip(edges, kappas)],
    }
    json_path = f'{OUT_DIR}/copurchase_ricci.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n✅ Saved: {json_path}')

    # Verdict
    write_verdict(m, len(edges), len(nodes))
    print(f'✅ Saved: {OUT_DIR}/verdict.md')
    print('\nDone.')


def write_verdict(m, n_edges, n_nodes):
    path = f'{OUT_DIR}/verdict.md'
    mean_k = m['mean']
    pos_frac = m['frac_pos']
    neg_frac = m['frac_neg']
    zero_frac = m['frac_zero']

    if mean_k < -0.05:
        interp = '负曲率（树状/双曲倾向）'
        rec = '深层宜用双曲几何（H-E-E-E）'
        narrative = (
            f'Copurchase 图 Ollivier-Ricci κ_mean={mean_k:+.4f}，呈负曲率。'
            f'共购关系具有"星形/树状"拓扑特征，适合双曲空间表达。'
            f'这为 H-E-E-E 的深层双曲分量提供了图结构层面的依据。'
        )
    elif mean_k > 0.05:
        interp = '正曲率（团状/球面倾向）'
        rec = '深层用球面几何（不推荐，双曲在这里可能价值有限）'
        narrative = (
            f'Copurchase 图 κ_mean={mean_k:+.4f}，呈正曲率。'
            f'共购关系趋向局部聚集而非树状层级，正曲率对推荐价值有限。'
        )
    else:
        interp = '近零曲率（平坦/欧氏倾向）'
        rec = '深层用欧氏几何即可'
        narrative = (
            f'Copurchase 图 κ_mean={mean_k:+.4f}，近乎平坦。'
            f'共购关系无显著曲率偏好，标准欧氏几何已充分近似。'
            f'与转移图结论一致：图结构本身是平坦的，深层几何选择的影响有限。'
        )

    lines = [
        '# Task 376 Verdict: Copurchase 图 Ollivier-Ricci 曲率',
        '',
        '## 设计',
        f'- **图**: {n_edges} 边, {n_nodes} 节点',
        f'- **方法**: Ollivier-Ricci（均匀邻居分布 + 总变差距离）',
        '',
        '## 曲率统计',
        '',
        f'| 指标 | 值 |',
        f'|------|-----|',
        f'| Mean κ | {m["mean"]:+.6f} |',
        f'| Median κ | {m["median"]:+.6f} |',
        f'| Std κ | {m["std"]:.6f} |',
        f'| κ > 0 | {pos_frac*100:.1f}% |',
        f'| κ < 0 | {neg_frac*100:.1f}% |',
        f'| κ ≈ 0 | {zero_frac*100:.1f}% |',
        f'| Range | [{m["min"]:+.4f}, {m["max"]:+.4f}] |',
        '',
        '## 解读',
        f'**整体曲率**: {interp}',
        f'**推荐几何**: {rec}',
        f'**直觉**: {narrative}',
        '',
        '## 转移图 vs Copurchase 图曲率对照',
        '',
        '| 图 | Mean κ | 解读 | 推荐几何 |',
        '|----|----------|------|------------|',
        f'| 转移图 | ≈0.000 | 近零平坦 | 欧氏 |',
        f'| Copurchase | {mean_k:+.4f} | {interp} | {rec} |',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
