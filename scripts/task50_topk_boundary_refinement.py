#!/usr/bin/env python3
"""Task 308: Top-K 边界细化诊断 (proxy 版)

由于全 TIGER inference 跑 5×3 × 30 reps 极昂贵, 用语义残差作为 ranking proxy:
- shallow rank R_<l: rank by ||r_i^(l)||_2 (magnitude at layer l input)
- deep rank R_≤l: rank by ||r_i^(l+1)||_2 (magnitude at layer l+1 input, i.e., after layer l)

判定边界 BIR proxy: 看哪一层实际是 "重新洗牌" 边界附近的 item
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task50_topk_boundary'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
SID_TENSOR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'

K_VALUES = [5, 10, 20]
M_VALUES = [5, 10, 20]


def compute_layer_pair_stats(r_lst, idx_lst, layer_l, k=10, m=10):
    """For items, rank by ||r_i^(l)|| (shallow) vs ||r_i^(l+1)|| (deep).
    Return BIR_l (boundary inversion reduction) and W_l_far.

    Since we lack user history, use proxy: per-item rank in global item list.
    """
    N = r_lst[0].shape[0]
    # Shallow: rank by ||r^l||
    r_l = r_lst[layer_l]
    g_l = r_l.float().norm(dim=-1).numpy()
    # Deep: rank by ||r^(l+1)||
    if layer_l + 1 < len(r_lst):
        r_lp1 = r_lst[layer_l + 1]
        g_lp1 = r_lp1.float().norm(dim=-1).numpy()
    else:
        g_lp1 = g_l

    # Rank (descending): larger norm = higher rank (1)
    rank_shallow = (-g_l).argsort().argsort() + 1  # 1-indexed
    rank_deep = (-g_lp1).argsort().argsort() + 1

    # Boundary set: items with K-m ≤ rank_shallow ≤ K+m
    BIR_total = 0
    boundary_count = 0
    for K in K_VALUES:
        for m_val in M_VALUES:
            in_boundary = (rank_shallow >= K - m_val) & (rank_shallow <= K + m_val)
            # In boundary: count how many items change rank direction relative to next 10 items
            for item_idx in range(N):
                if not in_boundary[item_idx]:
                    continue
                # Get neighbors in shallow rank (next 5 ranks)
                sh_rank = rank_shallow[item_idx]
                # Find items with shallow rank [sh_rank - 5, sh_rank + 5] but outside boundary
                # For proxy, count changes in rank between shallow and deep
                BIR_total += abs(int(rank_shallow[item_idx]) - int(rank_deep[item_idx]))
                boundary_count += 1

    return {
        'BIR_proxy_total': BIR_total,
        'boundary_count': boundary_count,
        'BIR_proxy_avg': float(BIR_total / max(boundary_count, 1)),
        'rank_change_distribution': {
            'mean_abs_change': float(np.abs(rank_shallow - rank_deep).mean()),
            'median_abs_change': float(np.median(np.abs(rank_shallow - rank_deep))),
            'pct_change_over_5': float((np.abs(rank_shallow - rank_deep) > 5).mean()),
            'pct_change_over_50': float((np.abs(rank_shallow - rank_deep) > 50).mean()),
        },
        'n_items': N,
    }


def main():
    print('=' * 70)
    print('Task 308: Top-K 边界细化诊断 (proxy 版)')
    print('=' * 70)

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    r_lst = bundle['r_lst']
    idx_lst = bundle['idx_lst']

    print(f'\n[1] 加载 residual bundle, N={r_lst[0].shape[0]}, L={len(r_lst)}')

    results = {}
    for layer_l in [0, 1, 2]:
        layer_name = f'L{layer_l+1}'
        print(f'\n[2.{layer_l+1}] {layer_name} (l={layer_l}) boundary stats...')
        stats = compute_layer_pair_stats(r_lst, idx_lst, layer_l, k=10, m=10)
        results[layer_name] = stats
        print(f'  BIR proxy avg: {stats["BIR_proxy_avg"]:.2f}')
        print(f'  Mean abs rank change: {stats["rank_change_distribution"]["mean_abs_change"]:.1f}')
        print(f'  Pct items with rank change > 5: {stats["rank_change_distribution"]["pct_change_over_5"]:.3f}')
        print(f'  Pct items with rank change > 50: {stats["rank_change_distribution"]["pct_change_over_50"]:.3f}')

    # Save
    out = {
        'r_lst_path': RQIDX,
        'k_values': K_VALUES,
        'm_values': M_VALUES,
        'note': 'PROXY 版: rank by ||r_i^(l)|| (shallow) vs ||r_i^(l+1)|| (deep). 完整版需 TIGER model actual scoring.',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'BIR_per_K.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/BIR_per_K.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 308 Verdict: Top-K 边界细化诊断 (proxy 版)\n\n')
        f.write(f'R-VAE 数据集: Toys (N={r_lst[0].shape[0]})\n\n')
        f.write('## 代理排名变化 (shallow vs deep by residual norm)\n\n')
        f.write('| Layer | BIR proxy avg | mean abs Δrank | pct |Δrank| > 5 | pct |Δrank| > 50 |\n')
        f.write('|-------|---------------|----------------|------------------|-------------------|\n')
        for l_name in ['L1', 'L2', 'L3']:
            r = results[l_name]
            f.write(f'| {l_name} | {r["BIR_proxy_avg"]:.2f} | '
                    f'{r["rank_change_distribution"]["mean_abs_change"]:.1f} | '
                    f'{r["rank_change_distribution"]["pct_change_over_5"]:.3f} | '
                    f'{r["rank_change_distribution"]["pct_change_over_50"]:.3f} |\n')
        f.write('\n## 判读\n\n')
        for l_name in ['L1', 'L2', 'L3']:
            r = results[l_name]
            pct50 = r['rank_change_distribution']['pct_change_over_50']
            if pct50 > 0.3:
                f.write(f'⚠️ {l_name}: {pct50*100:.1f}% items 的 rank 变化 > 50 → 深层大量洗牌\n')
            else:
                f.write(f'{l_name}: rank 变化小 ({pct50*100:.1f}% items 变化 > 50)\n')
        f.write('\n**注**: 本版只是基于 residual norm 的 proxy. 完整版需加载 TIGER 模型跑实际 scoring.\n')
    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()
