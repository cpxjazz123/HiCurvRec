#!/usr/bin/env python3
"""Task 301: 行为纯度增量 (3 简化 B, ΔH_l, BVR_l, CR vs BVR)

由于 Toys 单数据集无完整 user history, 使用 metadata proxy:
- cat_sub (24 类细分类)
- brand (item brand, ~700 unique)
- cat_top (大类, 6 类)

计算:
- H(B | Z_<l) 和 H(B | Z_≤l)
- ΔH_l^B = H(B|Z_<l) - H(B|Z_≤l)  (positive = layer l helps)
- BVR_l = 1 - V_l / V_<l
- 与 task42 CR_l 对比
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task43_behavior_purity'
os.makedirs(OUT_DIR, exist_ok=True)

SID_TENSOR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'
METADATA = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
TASK300_RESULT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task42_entropy_branching/entropy_per_layer.json'


def entropy_of_partition(B_per_partition):
    """B_per_partition: dict {partition_key: dict {B_value: count}}.
    Returns H(B|partition) weighted average.
    """
    total_items = sum(sum(bc.values()) for bc in B_per_partition.values())
    if total_items == 0:
        return 0.0, 0
    h_total = 0.0
    for pkey, bc in B_per_partition.items():
        n_p = sum(bc.values())
        if n_p == 0:
            continue
        h_p = 0.0
        for b, c in bc.items():
            if c > 0:
                p = c / n_p
                h_p -= p * np.log(p + 1e-12)
        # weight by P(p) = n_p / total_items
        h_total += (n_p / total_items) * h_p
    return h_total, total_items


def within_partition_variance(B_per_partition):
    """Compute within-partition variance of B: V = sum_p P(p) * Var(B | p).
    Assumes B is treated as categorical and variance is Gini-style.
    """
    total_n = sum(sum(bc.values()) for bc in B_per_partition.values())
    if total_n == 0:
        return 0.0
    var_total = 0.0
    for pkey, bc in B_per_partition.items():
        n_p = sum(bc.values())
        if n_p == 0:
            continue
        # Compute fraction of majority B class
        max_c = max(bc.values())
        # impurity = fraction of non-majority
        impurity = 1 - max_c / n_p
        # weight by P(p)
        var_total += (n_p / total_n) * impurity
    return float(var_total)


def compute_for_B(B_per_item, Z, layer_l):
    """For layer l:
    - Partition items by Z[<l+1] (using prefix + current z[l])
    - Compute H(B | Z_<l) and H(B | Z_≤l) and within-variance
    """
    N = len(B_per_item)
    if layer_l == 0:
        # All in one partition for prefix
        Z_lt_groups = {('all',): list(range(N))}
        Z_l_groups = defaultdict(list)
        for i in range(N):
            Z_l_groups[(int(Z[0, i].item()),)].append(i)
    else:
        # Group by prefix Z[<l]
        Z_lt_groups = defaultdict(list)
        for i in range(N):
            key = tuple(int(Z[k, i].item()) for k in range(layer_l))
            Z_lt_groups[key].append(i)
        # Group by Z[≤l] = prefix + current z[l]
        Z_l_groups = defaultdict(list)
        for i in range(N):
            key = tuple(int(Z[k, i].item()) for k in range(layer_l + 1))
            Z_l_groups[key].append(i)

    # Build B counts per partition for H(B|Z_<l)
    bc_lt = {}
    for pkey, items in Z_lt_groups.items():
        bc = defaultdict(int)
        for i in items:
            bc[B_per_item[i]] += 1
        bc_lt[pkey] = bc

    bc_l = {}
    for pkey, items in Z_l_groups.items():
        bc = defaultdict(int)
        for i in items:
            bc[B_per_item[i]] += 1
        bc_l[pkey] = bc

    # H(B|Z_<l)
    h_lt, n_lt = entropy_of_partition(bc_lt)
    h_l, n_l = entropy_of_partition(bc_l)

    # Within-variance
    v_lt = within_partition_variance(bc_lt)
    v_l = within_partition_variance(bc_l)

    delta_h = h_lt - h_l  # positive = layer helps
    bvr = 1 - v_l / (v_lt + 1e-12)

    return {
        'h_B_given_Z_lt_l': h_lt,
        'h_B_given_Z_le_l': h_l,
        'delta_H': float(delta_h),
        'V_Z_lt_l': v_lt,
        'V_Z_le_l': v_l,
        'BVR_l': float(bvr),
        'n_partitions_before': len(Z_lt_groups),
        'n_partitions_after': len(Z_l_groups),
    }


def main():
    print('=' * 70)
    print('Task 301: 行为纯度增量 (3 B 类型, ΔH, BVR, CR vs BVR)')
    print('=' * 70)

    # Load SID tensor
    Z_full = torch.load(SID_TENSOR, map_location='cpu', weights_only=False)
    Z = Z_full.t().long() if Z_full.shape[1] == 4 else Z_full.long()
    N = Z.shape[1]
    print(f'\n[1] 加载 SID tensor: shape={tuple(Z.shape)}, N={N}')

    # Load metadata
    print(f'\n[2] 加载 metadata from {METADATA}')
    m = json.load(open(METADATA))
    assert len(m) == N, f'metadata N={len(m)} != SID N={N}'

    # Build B distributions from metadata
    print(f'\n[3] 构造 B 分布...')
    # cat_sub, brand, cat_top
    B_dict = {}
    for B_name in ['cat_sub', 'brand', 'cat_top']:
        B_per_item = [m[str(i)][B_name] for i in range(N)]
        unique = set(B_per_item)
        # Convert to integer codes for efficiency
        unique_list = sorted(unique)
        B_int = [unique_list.index(b) for b in B_per_item]
        B_dict[B_name] = {
            'values': B_int,
            'unique_count': len(unique),
            'n_per_class': {u: B_per_item.count(u) for u in unique_list[:5]},  # preview
        }
        print(f'  {B_name}: {len(unique)} unique values, 样本分布 = {B_dict[B_name]["n_per_class"]}')

    # Per-layer × per-B stats
    results = {}
    for B_name in ['cat_sub', 'brand', 'cat_top']:
        results[B_name] = {}
        B_int = B_dict[B_name]['values']
        for layer_l in [0, 1, 2]:
            stats = compute_for_B(B_int, Z, layer_l)
            results[B_name][f'l={layer_l+1}'] = stats
            print(f'\n  [{B_name}, l={layer_l+1}] ΔH={stats["delta_H"]:.4f}, '
                  f'BVR={stats["BVR_l"]:.4f}, h_lt={stats["h_B_given_Z_lt_l"]:.3f}, '
                  f'h_l={stats["h_B_given_Z_le_l"]:.3f}')

    # Load task42 CR results
    cr_data = {}
    if os.path.exists(TASK300_RESULT):
        t300 = json.load(open(TASK300_RESULT))
        for l_name, stats in t300['results'].items():
            cr_data[l_name] = stats['cr_l']
        print(f'\n[4] 加载 task42 CR: {cr_data}')

    # Save
    out = {
        'n_items': N,
        'sid_tensor_path': SID_TENSOR,
        'metadata_path': METADATA,
        'note': '简化版: 仅 cat_sub/brand/cat_top 替代 5 个 B; 用户历史/next-item 需 user logs, 暂未引入',
        'B_used': list(B_dict.keys()),
        'results': results,
        'cr_per_layer': cr_data,
    }
    with open(os.path.join(OUT_DIR, 'delta_H_per_B.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/delta_H_per_B.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 301 Verdict: 行为纯度增量\n\n')
        f.write('数据集: Toys (N=11924), 简化为 cat_sub/brand/cat_top 三个 B 目标\n')
        f.write('(完整 task43 需 user history 取前序/后继/共现, 暂以 metadata 代理)\n\n')
        f.write('## 各 B × 各层统计\n\n')
        f.write('| B | Layer | ΔH | BVR | h(Z_<l) | h(Z_≤l) |\n')
        f.write('|---|-------|-----|-----|---------|----------|\n')
        for B_name in ['cat_sub', 'brand', 'cat_top']:
            for layer_l in [0, 1, 2]:
                stats = results[B_name][f'l={layer_l+1}']
                f.write(f'| {B_name} | L{layer_l+1} | {stats["delta_H"]:.4f} | '
                        f'{stats["BVR_l"]:.4f} | {stats["h_B_given_Z_lt_l"]:.3f} | '
                        f'{stats["h_B_given_Z_le_l"]:.3f} |\n')
        f.write('\n## CR vs BVR 对比\n\n')
        f.write('| Layer | CR_l (item区分) | BVR_l (cat_sub) | BVR_l (brand) | BVR_l (cat_top) |\n')
        f.write('|-------|-----------------|------------------|----------------|------------------|\n')
        for layer_l in [1, 2, 3]:
            l_name = f'L{layer_l}'
            cr = cr_data.get(l_name, 'N/A')
            bvr_sub = results['cat_sub'][f'l={layer_l}']['BVR_l']
            bvr_brand = results['brand'][f'l={layer_l}']['BVR_l']
            bvr_top = results['cat_top'][f'l={layer_l}']['BVR_l']
            f.write(f'| {l_name} | {cr:.4f} | {bvr_sub:.4f} | {bvr_brand:.4f} | {bvr_top:.4f} |\n')
        f.write('\n## 判读\n\n')
        # Compute trends
        for B_name in ['cat_sub', 'brand', 'cat_top']:
            delta_hs = [results[B_name][f'l={i+1}']['delta_H'] for i in range(3)]
            bvrs = [results[B_name][f'l={i+1}']['BVR_l'] for i in range(3)]
            f.write(f'### {B_name}\n')
            f.write(f'- ΔH 跨层: L1={delta_hs[0]:.4f}, L2={delta_hs[1]:.4f}, L3={delta_hs[2]:.4f}\n')
            f.write(f'- BVR 跨层: L1={bvrs[0]:.4f}, L2={bvrs[1]:.4f}, L3={bvrs[2]:.4f}\n')
            if delta_hs[2] < 0.01:
                f.write(f'  ⚠️ L3 ΔH 接近 0 → 该层对该 B 几乎不提供新区分\n')
            f.write('\n')

        f.write('### CR vs BVR 关键判据\n')
        crs = [cr_data.get(f'L{i+1}', 0) for i in range(3)]
        if all(c > 0.9 for c in crs[:3]) and all(results['cat_sub'][f'l={i+1}']['BVR_l'] < 0.3 for i in range(3)):
            f.write('⚠️ **CR 很高 (item 区分强) 但 BVR 很低 (行为区分弱)**\n')
            f.write('→ 深层 SID 主要是 item ID 区分, **不是行为区分**\n')
            f.write('→ 推荐任务需要的"行为区分能力"未被该编码捕获\n')
        else:
            f.write('CR 和 BVR 大致同步 (跨层变化方向一致)\n')
    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()
