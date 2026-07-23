#!/usr/bin/env python3
"""Task 362: 第一层码字结构保留度分析

核心问题：A vs B 判定
- 可能性 A: 第一层码字已经把结构学进去了，残差无结构是良性的任务完成
- 可能性 B: 第一层码字分配本身就没保留结构，问题在第一层

实验设计：
1. 对每个 item，从 RQ-VAE 拿第一层码字 index
2. 同 cat_sub item 是否倾向于分到相同/相邻码字？
3. Mantel test: d_codeword vs d_tree
   - d_codeword(i,j) = 0 if idx[i]==idx[j], ||C[idx[i]] - C[idx[j]]|| otherwise
4. kNN 结构保留度 (用 d_codeword 找最近邻)
5. 跨算法对照: A_RQ_VAE vs C_GSRQ (task15 三方)

判据:
- 若 Mantel ρ > 0.3 或同 cat_sub 码字集中度高 → 支持可能性 A
- 若 Mantel ρ ≈ 0 且 hit rate ≈ chance → 支持可能性 B（关键发现）
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from collections import defaultdict
from scipy.spatial.distance import cdist

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa: F401

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task362_l1_codeword_structure'
os.makedirs(OUT_DIR, exist_ok=True)

ITEMS_DIR = f'{GRID}/data/amazon_data/toys/items'
TRAIN_DIR = f'{GRID}/data/amazon_data/toys/training'
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

ALGOS = {
    'A_RQ_VAE': f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'C_GSRQ': f'{GRID}/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
}

sys.path.insert(0, f'{GRID}/task_artifacts/scripts')
from task351_residual_structure import (
    load_codebooks, forward_residual,
    load_item_metadata, build_d_tree, build_d_graph,
    mantel_test, gromov_delta, knn_structure_hit_rate,
)


def analyze_codeword_structure(idx, codebook, metadata, n_items, d_tree, d_graph,
                                sample_idx, n_perm=499, k=10):
    """对给定 L1 码字分配分析结构保留度。

    Args:
        idx: (n_items,) 第一层码字 index
        codebook: (K, D) 码本向量
        metadata: item_id -> {cat_sub, cat_top}
        n_items: 总 item 数
        d_tree: (n_items, n_items) taxonomy 距离矩阵
        d_graph: (n_items, n_items) co-purchase 距离矩阵
        sample_idx: 采样 item idx
        n_perm: Mantel 置换数
        k: kNN k

    Returns:
        dict: 多个结构指标
    """
    # 1. 构造 d_codeword: 同码字距离 0，不同码字 = ||C[i] - C[j]||
    cb = codebook.numpy() if torch.is_tensor(codebook) else codebook
    cb_dist = cdist(cb, cb, metric='euclidean').astype(np.float32)  # (K, K)

    # 构造 (n_items, n_items) 的 d_codeword
    # 高效方法：先转为 (n, K) one-hot，然后用 cb_dist
    # 但 K=256, n=11924 → 11924*256 = 3M 元素，可以
    n = n_items
    K = cb.shape[0]
    # 优化：对 sample 用索引构建
    idx_np = idx.cpu().numpy() if torch.is_tensor(idx) else idx

    # 对 sample 计算 d_codeword
    n_s = len(sample_idx)
    idx_s = idx_np[sample_idx]
    d_codeword_s = cb_dist[idx_s][:, idx_s]  # (n_s, n_s)

    # 2. Mantel test: d_codeword vs d_tree
    tree_mantel_r, tree_mantel_p = mantel_test(d_codeword_s, d_tree[sample_idx][:, sample_idx],
                                                n_perm=n_perm)
    graph_mantel_r, graph_mantel_p = mantel_test(d_codeword_s, d_graph[sample_idx][:, sample_idx],
                                                  n_perm=n_perm)

    # 3. 同 cat_sub item 的码字集中度
    cat_sub_arr = np.array([metadata.get(i, {}).get('cat_sub', 'Unknown') for i in range(n)])
    cat_top_arr = np.array([metadata.get(i, {}).get('cat_top', 'Unknown') for i in range(n)])

    # 跨 cat_sub 的码字熵 + 集中度
    cat_sub_unique = sorted(set(cat_sub_arr))
    cat_top_unique = sorted(set(cat_top_arr))

    # 对每个 cat_sub，分组看分配码字的分布
    cat_sub_codebook_stats = {}
    for cs in cat_sub_unique:
        mask = (cat_sub_arr == cs)
        if mask.sum() < 5:
            continue
        sub_idx = idx_np[mask]
        # 码字分布：每个码字的 item 数
        unique, counts = np.unique(sub_idx, return_counts=True)
        # 集中度：top-1 码字占比
        top1_ratio = counts.max() / counts.sum() if counts.sum() > 0 else 0
        # 集中度：top-3 码字占比
        top3_ratio = np.sort(counts)[-3:].sum() / counts.sum() if counts.sum() > 0 else 0
        # 码字熵 (越低 = 越集中)
        p = counts / counts.sum()
        entropy = -(p * np.log(p + 1e-12)).sum()
        # n_active_codes: 实际使用的码字数
        n_active = len(unique)
        cat_sub_codebook_stats[cs] = {
            'n_items': int(mask.sum()),
            'n_active_codes': n_active,
            'top1_ratio': float(top1_ratio),
            'top3_ratio': float(top3_ratio),
            'entropy': float(entropy),
            'unique_codes': unique.tolist()[:10],  # 截断
        }

    # 全局统计
    unique_all, counts_all = np.unique(idx_np, return_counts=True)
    n_active_global = len(unique_all)
    global_entropy = -(counts_all / counts_all.sum() * np.log(counts_all / counts_all.sum() + 1e-12)).sum()
    gini = _gini_coefficient(counts_all)

    # 4. kNN 结构保留度 (用 d_codeword)
    # 因为 d_codeword 是 sample × sample (n_s=1500)，可以 fit KNN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
    nn.fit(d_codeword_s)
    _, knn_idx = nn.kneighbors(d_codeword_s)
    knn_idx = knn_idx[:, 1:]  # 排除自身

    # 同 cat_sub 命中率
    cat_sub_s = cat_sub_arr[sample_idx]
    cat_top_s = cat_top_arr[sample_idx]
    tree_s = d_tree[sample_idx][:, sample_idx]
    graph_s = d_graph[sample_idx][:, sample_idx]

    hits_tree, total_tree = 0, 0
    for i in range(n_s):
        for jj in knn_idx[i]:
            total_tree += 1
            if tree_s[i, jj] < 1.0:  # 同 cat_sub (d_tree < 1.0)
                hits_tree += 1
    hit_rate_tree = hits_tree / total_tree

    # 随机基线（chance）：permuting d_tree
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_s)
    tree_s_shuf = tree_s[perm][:, perm]
    hits_chance, total_chance = 0, 0
    for i in range(n_s):
        for jj in knn_idx[i]:
            total_chance += 1
            if tree_s_shuf[i, jj] < 1.0:
                hits_chance += 1
    chance_tree = hits_chance / total_chance

    # 共购 hit rate
    hits_graph, total_graph = 0, 0
    for i in range(n_s):
        for jj in knn_idx[i]:
            total_graph += 1
            if graph_s[i, jj] < 1.0:  # 共购过
                hits_graph += 1
    hit_rate_graph = hits_graph / total_graph

    perm_g = rng.permutation(n_s)
    graph_s_shuf = graph_s[perm_g][:, perm_g]
    hits_chance_g, _ = 0, 0
    for i in range(n_s):
        for jj in knn_idx[i]:
            if graph_s_shuf[i, jj] < 1.0:
                hits_chance_g += 1
    chance_graph = hits_chance_g / total_graph

    return {
        'mantel_tree': {'r': float(tree_mantel_r), 'p': float(tree_mantel_p)},
        'mantel_graph': {'r': float(graph_mantel_r), 'p': float(graph_mantel_p)},
        'global_codebook_stats': {
            'n_active_codes': n_active_global,
            'K': K,
            'global_entropy': float(global_entropy),
            'gini': float(gini),
            'max_count': int(counts_all.max()),
            'min_count': int(counts_all.min()),
        },
        'per_cat_sub_stats': cat_sub_codebook_stats,
        'knn_tree_hit': float(hit_rate_tree),
        'knn_tree_chance': float(chance_tree),
        'knn_graph_hit': float(hit_rate_graph),
        'knn_graph_chance': float(chance_graph),
        'lift_tree': float(hit_rate_tree - chance_tree),
        'lift_graph': float(hit_rate_graph - chance_graph),
    }


def _gini_coefficient(counts):
    """Gini 系数（码字使用不均衡度）。"""
    c = np.sort(np.array(counts, dtype=np.float64))
    n = len(c)
    if c.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * c).sum() / (n * c.sum())) - (n + 1) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=1500)
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--n-perm', type=int, default=499)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    print('=' * 70)
    print('Task 362: L1 码字结构保留度 (A vs B 关键判定)')
    print('=' * 70)

    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    if emb.shape[0] == 2048 and emb.shape[1] == 11924:
        emb = emb.T
    n_items = emb.shape[0]
    print(f'embedding shape: {emb.shape}')

    metadata = load_item_metadata(n_items)
    d_tree, n_cat_sub, n_cat_top = build_d_tree(metadata, n_items)
    d_graph, n_nonzero_graph = build_d_graph(n_items, n_users=args.n_users)

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_items, min(args.n_sample, n_items), replace=False)

    results = {
        'task': 'task362_l1_codeword_structure',
        'method': 'Mantel d_codeword vs d_tree + 同 cat_sub 码字集中度',
        'n_items': n_items,
        'n_sample': len(sample_idx),
        'n_perm': args.n_perm,
        'k': args.k,
        'n_cat_sub_unique': n_cat_sub,
        'n_cat_top_unique': n_cat_top,
        'algos': {},
    }

    for algo, ckpt_path in ALGOS.items():
        print(f'\n=== Algorithm: {algo} ({ckpt_path}) ===')
        codebooks, gains, has_gains, normalize, L = load_codebooks(ckpt_path)
        print(f'  L={L}, normalize_residuals={normalize}, has_gains={has_gains}')

        r_lst, q_lst, idx_lst = forward_residual(emb, codebooks, gains)
        # L1 码字 index
        l1_idx = idx_lst[0]
        l1_codebook = codebooks[0]

        # 检查 gain (如果 has_gains, codebook 已经乘了 gain)
        if gains is not None and gains[0] is not None:
            l1_codebook_eff = l1_codebook * gains[0].unsqueeze(-1)
        else:
            l1_codebook_eff = l1_codebook

        K = l1_codebook.shape[0]
        print(f'  K (codebook size): {K}')

        # L1 分配统计
        l1_unique, l1_counts = torch.unique(l1_idx, return_counts=True)
        print(f'  L1 active codes: {len(l1_unique)}/{K}')
        print(f'  L1 count max/min/mean: {l1_counts.max()}/{l1_counts.min()}/{l1_counts.float().mean():.1f}')

        stats = analyze_codeword_structure(
            l1_idx, l1_codebook_eff, metadata, n_items, d_tree, d_graph,
            sample_idx, n_perm=args.n_perm, k=args.k)

        print(f'\n  Mantel.tree: {stats["mantel_tree"]["r"]:+.4f} (p={stats["mantel_tree"]["p"]:.4f})')
        print(f'  Mantel.graph: {stats["mantel_graph"]["r"]:+.4f} (p={stats["mantel_graph"]["p"]:.4f})')
        print(f'  kNN.tree hit: {stats["knn_tree_hit"]:.4f} (chance {stats["knn_tree_chance"]:.4f}, lift {stats["lift_tree"]:+.4f})')
        print(f'  kNN.graph hit: {stats["knn_graph_hit"]:.4f} (chance {stats["knn_graph_chance"]:.4f}, lift {stats["lift_graph"]:+.4f})')
        print(f'  Gini: {stats["global_codebook_stats"]["gini"]:.4f}')
        print(f'  Global entropy: {stats["global_codebook_stats"]["global_entropy"]:.4f}')

        # 打印 cat_sub 集中度 Top-5
        print(f'\n  cat_sub 集中度 (top-5 by n_items):')
        cs_stats = sorted(stats['per_cat_sub_stats'].items(),
                          key=lambda x: -x[1]['n_items'])[:5]
        for cs, st in cs_stats:
            print(f'    {cs[:30]:<30}: n={st["n_items"]}, top1={st["top1_ratio"]:.3f}, '
                  f'top3={st["top3_ratio"]:.3f}, entropy={st["entropy"]:.2f}, n_active={st["n_active_codes"]}')

        results['algos'][algo] = {
            'ckpt_path': ckpt_path,
            'L': L,
            'K': K,
            'normalize_residuals': normalize,
            'has_gains': has_gains,
            'l1_active_codes': len(l1_unique),
            'l1_count_max': int(l1_counts.max()),
            'l1_count_min': int(l1_counts.min()),
            'l1_count_mean': float(l1_counts.float().mean()),
            **stats,
        }

    json_path = f'{OUT_DIR}/l1_codeword_structure.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n✅ Saved: {json_path}')

    write_verdict(results, f'{OUT_DIR}/verdict.md')
    print(f'✅ Saved: {OUT_DIR}/verdict.md')

    print('\n' + '=' * 70)
    print('SUMMARY: A vs B 关键判定')
    print('=' * 70)
    for algo in ['A_RQ_VAE', 'C_GSRQ']:
        d = results['algos'][algo]
        print(f'  {algo}:')
        print(f'    Mantel.tree ρ: {d["mantel_tree"]["r"]:+.4f} (p={d["mantel_tree"]["p"]:.4f})')
        print(f'    kNN.tree hit: {d["knn_tree_hit"]:.4f} (chance {d["knn_tree_chance"]:.4f}, lift {d["lift_tree"]:+.4f})')
        print(f'    kNN.graph hit: {d["knn_graph_hit"]:.4f} (chance {d["knn_graph_chance"]:.4f}, lift {d["lift_graph"]:+.4f})')
        print(f'    Gini: {d["global_codebook_stats"]["gini"]:.4f}')
    print('=' * 70)


def write_verdict(results, path):
    a = results['algos']['A_RQ_VAE']
    c = results['algos']['C_GSRQ']

    lines = [
        '# Task 362 Verdict: L1 码字结构保留度 (A vs B 关键判定)',
        '',
        '## 设计',
        '',
        '- **核心问题**：task351/357 只测了"残差里有没有结构"，无法区分 A vs B',
        '- **A (良性)**：第一层码字已经把结构学进去 → 残差无结构是任务完成',
        '- **B (恶性)**：第一层码字分配本身就没保留结构 → 问题在第一层',
        '- **方法**：',
        '  - 测 L1 码字 index 分配：同 cat_sub item 是否分到相同/相邻码字',
        '  - Mantel: d_codeword(i,j) vs d_tree(i,j)',
        '    - d_codeword(i,j) = 0 if idx[i]==idx[j], ||C[idx[i]]-C[idx[j]]|| otherwise',
        '  - kNN 用 d_codeword 找最近邻，看同 cat_sub hit rate vs 随机 chance',
        '  - 跨 cat_sub 集中度 (top-1 ratio, entropy)',
        '',
        '## 现象',
        '',
        '### 关键指标对照',
        '',
        '| 指标 | A_RQ_VAE | C_GSRQ | 解读 |',
        '|------|----------|--------|------|',
        f'| Mantel.tree ρ | {a["mantel_tree"]["r"]:+.4f} (p={a["mantel_tree"]["p"]:.3f}) | {c["mantel_tree"]["r"]:+.4f} (p={c["mantel_tree"]["p"]:.3f}) | d_codeword 与 taxonomy 相关性 |',
        f'| kNN.tree hit | {a["knn_tree_hit"]:.4f} (chance {a["knn_tree_chance"]:.4f}) | {c["knn_tree_hit"]:.4f} (chance {c["knn_tree_chance"]:.4f}) | L1 码字分配后同类目命中率 |',
        f'| kNN.graph hit | {a["knn_graph_hit"]:.4f} (chance {a["knn_graph_chance"]:.4f}) | {c["knn_graph_hit"]:.4f} (chance {c["knn_graph_chance"]:.4f}) | L1 码字分配后共购命中率 |',
        f'| Gini | {a["global_codebook_stats"]["gini"]:.4f} | {c["global_codebook_stats"]["gini"]:.4f} | 码字使用不均衡度 |',
        f'| L1 active codes | {a["l1_active_codes"]}/{a["K"]} | {c["l1_active_codes"]}/{c["K"]} | 码字利用率 |',
        '',
        '### Per-cat_sub 集中度 Top-5 (A_RQ_VAE)',
        '',
        '| cat_sub | n_items | top1_ratio | top3_ratio | entropy | n_active_codes |',
        '|---------|---------|------------|------------|---------|----------------|',
    ]
    cs_stats_a = sorted(a['per_cat_sub_stats'].items(), key=lambda x: -x[1]['n_items'])[:5]
    for cs, st in cs_stats_a:
        lines.append(
            f'| {cs[:30]} | {st["n_items"]} | {st["top1_ratio"]:.3f} | '
            f'{st["top3_ratio"]:.3f} | {st["entropy"]:.2f} | {st["n_active_codes"]} |'
        )

    lines.append('')
    lines.append('### Per-cat_sub 集中度 Top-5 (C_GSRQ)')
    lines.append('')
    lines.append('| cat_sub | n_items | top1_ratio | top3_ratio | entropy | n_active_codes |')
    lines.append('|---------|---------|------------|------------|---------|----------------|')
    cs_stats_c = sorted(c['per_cat_sub_stats'].items(), key=lambda x: -x[1]['n_items'])[:5]
    for cs, st in cs_stats_c:
        lines.append(
            f'| {cs[:30]} | {st["n_items"]} | {st["top1_ratio"]:.3f} | '
            f'{st["top3_ratio"]:.3f} | {st["entropy"]:.2f} | {st["n_active_codes"]} |'
        )

    lines.extend([
        '',
        '## 结论 — A vs B 关键判定',
        '',
    ])

    # 判定逻辑
    a_mantel = a['mantel_tree']['r']
    c_mantel = c['mantel_tree']['r']
    a_lift_tree = a['lift_tree']
    c_lift_tree = c['lift_tree']
    a_lift_graph = a['lift_graph']
    c_lift_graph = c['lift_graph']

    a_top1_avg = np.mean([s['top1_ratio'] for s in a['per_cat_sub_stats'].values()])

    if a_mantel > 0.3 and a_lift_tree > 0.05:
        lines.append(f'**A_RQ_VAE 支持可能性 A（良性）**：Mantel.tree={a_mantel:+.4f} > 0.3, kNN.tree lift={a_lift_tree:+.4f}')
        lines.append('→ 第一层码字分配确实把结构学进去了，残差无结构是良性的"任务完成"')
    elif a_mantel < 0.1 and a_lift_tree < 0.02:
        lines.append(f'**A_RQ_VAE 支持可能性 B（恶性）**：Mantel.tree={a_mantel:+.4f} < 0.1, kNN.tree lift={a_lift_tree:+.4f} ≈ 0')
        lines.append('→ 第一层码字分配本身就没把结构保留好 → 标准 RQ-VAE 硬分配机制的固有问题')
        lines.append('→ 这意味着 mixed-geometry / HRQ 解决的问题不是"深层无结构"，而是"第一层本身的结构保留"')
        lines.append('→ 论文叙事需要大改：differential geometry 应作用于第一层量化器，而不是深层')
    else:
        lines.append(f'**A_RQ_VAE 中间地带**：Mantel.tree={a_mantel:+.4f}, kNN.tree lift={a_lift_tree:+.4f}')
        lines.append('→ L1 码字保留部分结构，需要结合更多指标判定')

    lines.append('')

    if c_mantel > 0.3 and c_lift_tree > 0.05:
        lines.append(f'**C_GSRQ 支持可能性 A**：Mantel.tree={c_mantel:+.4f}, kNN.tree lift={c_lift_tree:+.4f}')
        lines.append('→ gain-shape 融合让 L1 码字保留更多 taxonomy 结构')
        lines.append('→ 这解释了 task357 发现（GSRQ 在 l=1 结构 +2.85x）')
    else:
        lines.append(f'**C_GSRQ 同样部分保留结构**：Mantel.tree={c_mantel:+.4f}')

    lines.extend([
        '',
        '## 论文叙事影响',
        '',
        '### 如果判定为 A（良性）',
        '- 残差无结构 = 任务完成，RQ-VAE 工作机制正确',
        '- mixed-geometry idea **优先级下降**（深层本就不该有结构）',
        '- 应关注 L1 本身的设计（cat_sub 集中度高的码字被学到）',
        '',
        '### 如果判定为 B（恶性）',
        '- L1 硬分配（argmin NN）机制本身不保留结构',
        '- mixed-geometry idea **应作用于 L1 而非深层**',
        '- 论文叙事：HRQ 不是"给深层减负"，而是"让 L1 量化器在双曲空间工作"',
        '- 设计方向：L1 应该用 hyperbolic K-means / soft assignment（替代硬 argmin）',
        '',
        '### task15 R@10 重读',
        f'- A_RQ_VAE R@10 = 0.09731（最高）',
        f'- C_GSRQ R@10 = 0.08572（最低）',
        f'- 若 L1 已包含结构（A 假设），A 赢 R@10 是因为 L1 结构捕获最有效',
        f'- 若 L1 没保留结构（B 假设），C_GSRQ R@10 输是因为 gain-shape 反而打乱了 L1 的最有效结构',
        '',
        '## 建议',
        '',
        '1. 把 L1 码字 index 直接作为 item embedding（替代 flan-t5-xl）训 TIGER，看 R@10 是否能达到 baseline',
        '   - 若 R@10 ≈ 0.09731：L1 码字已经捕获所有预测信号 → 支持 A',
        '   - 若 R@10 << 0.09731：L1 码字远不如原始 embedding → B',
        '2. 训 hyperbolic L1（hyperbolic K-means）vs 当前 L1，对照 R@10',
        '3. 对照 L2/L3 码字：跨层码字分配的 cat_sub 集中度是否递减？',
        '',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()