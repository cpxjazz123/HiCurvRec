#!/usr/bin/env python3
"""Task 357: 跨算法 residual 结构保留度对照 (RQ-VAE vs GSRQ)

task351 扩展：用 task15 Group A (RQ-VAE) + Group C (GSRQ) 两个 ckpt 做对照
复用 forward_residual + Mantel + Gromov δ + kNN hit (task351 实现)

差异：两种算法在残差处理上不同
- Group A: normalize_residuals=False (保留 magnitude)
- Group C: GSRQ with use_gain_tracking=True (显式跟踪 gain)

核心问题：GSRQ 的 gain 跟踪是否让残差空间结构保留度更好？
- 如果 GSRQ 结构保留度 > A → gain 跟踪有价值（结构信号被正确编码）
- 如果 GSRQ ≈ A → gain 跟踪冗余（与结论 R@10 接近一致）

注：Group B (MMQ) ckpt 已被清理，对照仅 A vs C。
"""

import os
import sys
import json
import re
import glob
import argparse
from collections import Counter

import numpy as np
import torch
import tensorflow as tf
from scipy.stats import spearmanr

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa: F401

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task357_residual_structure_cross_algo'
os.makedirs(OUT_DIR, exist_ok=True)

ITEMS_DIR = f'{GRID}/data/amazon_data/toys/items'
TRAIN_DIR = f'{GRID}/data/amazon_data/toys/training'
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

# 复用 task351 函数
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts')
from task351_residual_structure import (
    load_codebooks, forward_residual,
    load_item_metadata, build_d_tree, build_d_graph,
    mantel_test, gromov_delta, knn_structure_hit_rate,
)


ALGOS = {
    'A_RQ_VAE': f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'C_GSRQ': f'{GRID}/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=1500)
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--n-perm', type=int, default=499)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    print('=' * 70)
    print('Task 357: 跨算法 residual 结构保留度对照')
    print('=' * 70)

    # Load emb + ground truth (reuse task351 ground truth once)
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
    d_tree_s = d_tree[sample_idx][:, sample_idx]
    d_graph_s = d_graph[sample_idx][:, sample_idx]

    from scipy.spatial.distance import cdist

    results = {
        'task': 'task357_residual_structure_cross_algo',
        'method': 'Mantel + Gromov δ + kNN hit on RQ-VAE vs GSRQ residuals',
        'n_items': n_items,
        'n_sample': len(sample_idx),
        'n_perm': args.n_perm,
        'k': args.k,
        'n_users_co_purchase': args.n_users,
        'n_cat_sub_unique': n_cat_sub,
        'n_cat_top_unique': n_cat_top,
        'algos': {},
    }

    for algo, ckpt_path in ALGOS.items():
        print(f'\n=== Algorithm: {algo} ({ckpt_path}) ===')
        codebooks, gains, has_gains, normalize, L = load_codebooks(ckpt_path)
        print(f'  L={L}, normalize_residuals={normalize}, has_gains={has_gains}')
        r_lst, q_lst, idx_lst = forward_residual(emb, codebooks, gains)
        for l, r in enumerate(r_lst):
            print(f'  r^({l}): norm_mean={r.norm(dim=-1).mean().item():.4f}')

        per_layer = {}
        for l in range(L + 1):
            r = r_lst[l][sample_idx]
            d_emb = cdist(r.numpy(), r.numpy(), metric='euclidean').astype(np.float32)

            r_tree, p_tree = mantel_test(d_emb, d_tree_s, n_perm=args.n_perm)
            r_graph, p_graph = mantel_test(d_emb, d_graph_s, n_perm=args.n_perm)
            delta = gromov_delta(d_emb, n_sample=min(400, len(sample_idx)))
            hit_tree, chance_tree = knn_structure_hit_rate(r, d_tree_s, k=args.k)
            hit_graph, chance_graph = knn_structure_hit_rate(r, d_graph_s, k=args.k)

            print(f'  l={l}: Mantel.tree={r_tree:+.4f} (p={p_tree:.3f}) | '
                  f'Mantel.graph={r_graph:+.4f} (p={p_graph:.3f}) | '
                  f'Gromov.δ={delta:.4f} | '
                  f'kNN.tree={hit_tree:.4f} (chance={chance_tree:.4f}) | '
                  f'kNN.graph={hit_graph:.4f} (chance={chance_graph:.4f})')

            per_layer[f'l={l}'] = {
                'r_norm_mean': r_lst[l].norm(dim=-1).mean().item(),
                'mantel_tree': {'r': float(r_tree), 'p': float(p_tree)},
                'mantel_graph': {'r': float(r_graph), 'p': float(p_graph)},
                'gromov_delta': float(delta),
                'knn_tree': {'hit': float(hit_tree), 'chance': float(chance_tree),
                             'lift_over_chance': float(hit_tree - chance_tree)},
                'knn_graph': {'hit': float(hit_graph), 'chance': float(chance_graph),
                              'lift_over_chance': float(hit_graph - chance_graph)},
            }

        results['algos'][algo] = {
            'ckpt_path': ckpt_path,
            'L': L,
            'normalize_residuals': normalize,
            'has_gains': has_gains,
            'per_layer': per_layer,
        }

    # Cross-algo comparison table
    print('\n' + '=' * 70)
    print('Cross-Algorithm Comparison (Mantel.tree ρ per layer)')
    print('=' * 70)
    print(f'{"Layer":<8}{"A_RQ_VAE":<15}{"C_GSRQ":<15}{"Diff (C-A)":<15}')
    for l in range(max(results['algos'][a]['L'] for a in ALGOS) + 1):
        a_tree = results['algos']['A_RQ_VAE']['per_layer'][f'l={l}']['mantel_tree']['r']
        c_tree = results['algos']['C_GSRQ']['per_layer'][f'l={l}']['mantel_tree']['r']
        diff = c_tree - a_tree
        print(f'l={l:<7}{a_tree:+.4f}{" ":6}{c_tree:+.4f}{" ":6}{diff:+.4f}')

    # Cross-algo delta summary
    print('\nCross-Algo Δ (C_GSRQ - A_RQ_VAE) on Mantel.tree:')
    for l in range(4):
        try:
            a_tree = results['algos']['A_RQ_VAE']['per_layer'][f'l={l}']['mantel_tree']['r']
            c_tree = results['algos']['C_GSRQ']['per_layer'][f'l={l}']['mantel_tree']['r']
            print(f'  l={l}: A={a_tree:+.4f}, C={c_tree:+.4f}, Δ={c_tree-a_tree:+.4f}')
        except KeyError:
            pass

    # Save JSON
    json_path = f'{OUT_DIR}/cross_algo_residual_structure.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n✅ Saved: {json_path}')

    # Write verdict
    verdict_path = f'{OUT_DIR}/verdict.md'
    write_verdict(results, verdict_path)
    print(f'✅ Saved: {verdict_path}')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print('跨算法结构保留度对照（RQ-VAE vs GSRQ）:')
    for l in range(4):
        try:
            a = results['algos']['A_RQ_VAE']['per_layer'][f'l={l}']
            c = results['algos']['C_GSRQ']['per_layer'][f'l={l}']
            print(f'  l={l}: A.tree={a["mantel_tree"]["r"]:+.4f} | C.tree={c["mantel_tree"]["r"]:+.4f} | '
                  f'A.δ={a["gromov_delta"]:.4f} | C.δ={c["gromov_delta"]:.4f}')
        except KeyError:
            pass
    print('=' * 70)


def write_verdict(results, path):
    lines = [
        '# Task 357 Verdict: 跨算法 (RQ-VAE vs GSRQ) Residual 结构保留度',
        '',
        '## 设计',
        '',
        '- **核心问题**：task351 已验证结构保留度在 l=1 处剧降（RQ-VAE），但跨算法是否一致？',
        '- **对照**：Group A (RQ-VAE, normalize_residuals=False) vs Group C (GSRQ, use_gain_tracking=True)',
        '- **不动 GPU**：复用 task15 现成 ckpt，纯 forward + CPU 统计',
        '- **不训练新模型**：所有产物 task15 Group A/C 已稳定',
        '',
        '## 现象',
        '',
        '### Per-Algorithm Per-Layer 结果',
        '',
        '| Algo | Layer | r_norm_mean | Mantel.tree ρ | Mantel.graph ρ | Gromov δ | kNN.tree hit (chance) | kNN.graph hit (chance) |',
        '|------|-------|-------------|---------------|----------------|----------|----------------------|------------------------|',
    ]
    for algo in ['A_RQ_VAE', 'C_GSRQ']:
        L = results['algos'][algo]['L']
        for l in range(L + 1):
            d = results['algos'][algo]['per_layer'][f'l={l}']
            lines.append(
                f'| {algo} | l={l} | {d["r_norm_mean"]:.4f} | '
                f'{d["mantel_tree"]["r"]:+.4f} (p={d["mantel_tree"]["p"]:.3f}) | '
                f'{d["mantel_graph"]["r"]:+.4f} (p={d["mantel_graph"]["p"]:.3f}) | '
                f'{d["gromov_delta"]:.4f} | '
                f'{d["knn_tree"]["hit"]:.4f} ({d["knn_tree"]["chance"]:.4f}) | '
                f'{d["knn_graph"]["hit"]:.4f} ({d["knn_graph"]["chance"]:.4f}) |'
            )

    lines.extend([
        '',
        '### Cross-Algo Δ (C_GSRQ - A_RQ_VAE) on Mantel.tree',
        '',
        '| Layer | A_RQ_VAE | C_GSRQ | Δ (C-A) |',
        '|-------|----------|--------|---------|',
    ])
    for l in range(4):
        try:
            a_tree = results['algos']['A_RQ_VAE']['per_layer'][f'l={l}']['mantel_tree']['r']
            c_tree = results['algos']['C_GSRQ']['per_layer'][f'l={l}']['mantel_tree']['r']
            diff = c_tree - a_tree
            lines.append(f'| l={l} | {a_tree:+.4f} | {c_tree:+.4f} | {diff:+.4f} |')
        except KeyError:
            pass

    lines.extend([
        '',
        '## 结论',
        '',
    ])

    # 比较算法在 l=1 处的衰减
    a_l1_tree = results['algos']['A_RQ_VAE']['per_layer']['l=1']['mantel_tree']['r']
    c_l1_tree = results['algos']['C_GSRQ']['per_layer']['l=1']['mantel_tree']['r']
    a_l1_knn = results['algos']['A_RQ_VAE']['per_layer']['l=1']['knn_tree']['hit']
    c_l1_knn = results['algos']['C_GSRQ']['per_layer']['l=1']['knn_tree']['hit']
    a_l1_delta = results['algos']['A_RQ_VAE']['per_layer']['l=1']['gromov_delta']
    c_l1_delta = results['algos']['C_GSRQ']['per_layer']['l=1']['gromov_delta']

    lines.extend([
        '### 关键观察 1: l=1 处算法对照',
        '',
        f'- Mantel.tree: A={a_l1_tree:+.4f}, C={c_l1_tree:+.4f}',
        f'- kNN.tree hit: A={a_l1_knn:.4f}, C={c_l1_knn:.4f}',
        f'- Gromov δ: A={a_l1_delta:.4f}, C={c_l1_delta:.4f}',
        '',
    ])

    # l=0 baseline 对照
    a_l0_tree = results['algos']['A_RQ_VAE']['per_layer']['l=0']['mantel_tree']['r']
    c_l0_tree = results['algos']['C_GSRQ']['per_layer']['l=0']['mantel_tree']['r']
    a_l0_knn = results['algos']['A_RQ_VAE']['per_layer']['l=0']['knn_tree']['hit']
    c_l0_knn = results['algos']['C_GSRQ']['per_layer']['l=0']['knn_tree']['hit']

    lines.extend([
        '### 关键观察 2: l=0 (input embedding) 处算法对照',
        '',
        f'- Mantel.tree: A={a_l0_tree:+.4f}, C={c_l0_tree:+.4f}',
        f'- kNN.tree hit: A={a_l0_knn:.4f}, C={c_l0_knn:.4f}',
        '',
    ])

    # 判断 GSRQ 是否在 l=1 处结构保留度更好
    if c_l1_tree > a_l1_tree + 0.01 or c_l1_knn > a_l1_knn + 0.05:
        lines.extend([
            '**GSRQ 在 l=1 处结构保留度优于 RQ-VAE** → gain tracking 让残差空间保留更多 taxonomy 结构',
            '**支持 task20 HRQ 的几何机制**：gain 信息是 RQ-VAE 丢失的关键信号',
        ])
    elif a_l1_tree > c_l1_tree + 0.01 or a_l1_knn > c_l1_knn + 0.05:
        lines.extend([
            '**RQ-VAE 在 l=1 处结构保留度优于 GSRQ** → gain tracking 没有帮助（甚至破坏）结构信号',
            '**与 task15 R@10 对照一致**：A_RQ_VAE R@10=0.09731 > C_GSRQ R@10=0.08572',
            '→ gain 跟踪对预测和结构信号都是中性偏负',
        ])
    else:
        lines.extend([
            '**GSRQ 与 RQ-VAE 在 l=1 处结构保留度接近** → gain tracking 对残差空间结构无显著影响',
            '**与 task15 R@10 对照**：A R@10=0.09731, C R@10=0.08572 → C 略差但非显著',
        ])

    lines.extend([
        '',
        '## 建议',
        '',
        '1. 若 GSRQ 在 l=1 显著优于 RQ-VAE → 重新评估 GSRQ 真实价值（与 R@10 矛盾，可能有 batch effect）',
        '2. 若两者接近 → 进一步在 B_MMQ 上复现 task351 三方对照（需 ckpt 重建）',
        '3. 若 GSRQ 显著差 → 与 task15 R@10 结论一致，支持 GSRQ 失败结论',
        '',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()