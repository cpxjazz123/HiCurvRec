#!/usr/bin/env python3
"""Task 303: 精确 Sibling Ranking 实验 (结构分析 + rank 改善, CPU only)

定义 sibling set S_l(y_u) = {j : Z[j, <l] = Z[y_u, <l]} 即共享同 prefix 的物品集.
分析:
- sibling set 大小分布 (按 layer)
- 4 桶: [2,3], [4,10], [11,100], [100+]
- 平均 sibling 内多样性
- ΔRank (with vs without deep layer) 需要 TIGER model
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task45_sibling_ranking'
os.makedirs(OUT_DIR, exist_ok=True)

SID_TENSOR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'


def get_siblings(Z, target_idx, l):
    """Get sibling set S_l(target_idx) = {j : Z[j, <l] = Z[target_idx, <l]}"""
    if l == 0:
        # empty prefix: all items are siblings
        return list(range(Z.shape[1]))
    prefix = Z[:l, target_idx]
    siblings = []
    for j in range(Z.shape[1]):
        if torch.equal(Z[:l, j], prefix):
            siblings.append(j)
    return siblings


def main():
    print('=' * 70)
    print('Task 303: 精确 Sibling Ranking (结构分析, CPU only)')
    print('=' * 70)

    # Load SID tensor
    Z_full = torch.load(SID_TENSOR, map_location='cpu', weights_only=False)
    if Z_full.dim() == 2 and Z_full.shape[1] == 4:
        Z = Z_full.t().long()
    elif Z_full.dim() == 2 and Z_full.shape[0] == 4:
        Z = Z_full.long()
    else:
        raise ValueError(f'Unexpected SID shape: {Z_full.shape}')
    N = Z.shape[1]
    print(f'\n[1] 加载 SID tensor: shape={tuple(Z.shape)}, N={N}')

    # Define buckets
    buckets = [(2, 3), (4, 10), (11, 100), (101, 10**9)]
    bucket_names = ['[2,3]', '[4,10]', '[11,100]', '[101+]']

    results = {}
    for l in range(3):  # test layers L1, L2, L3 (index 0, 1, 2)
        layer_name = f'L{l+1}'
        print(f'\n[2.{l+1}] {layer_name} sibling 结构分析...')

        # Compute sibling set sizes for all items
        sibling_sizes = np.zeros(N, dtype=int)
        if l == 0:
            sibling_sizes[:] = N
        else:
            # group items by prefix
            Z_lt = Z[:l]
            prefixes = [tuple(int(Z_lt[k, i].item()) for k in range(l)) for i in range(N)]
            pfx_to_items = defaultdict(list)
            for i, p in enumerate(prefixes):
                pfx_to_items[p].append(i)
            for i in range(N):
                sibling_sizes[i] = len(pfx_to_items[prefixes[i]])

        # Distribution
        print(f'  sibling size: min={sibling_sizes.min()}, max={sibling_sizes.max()}, '
              f'mean={sibling_sizes.mean():.2f}, median={np.median(sibling_sizes):.1f}')

        # Bucket stats
        bucket_stats = {}
        bucket_masks = {}
        for name, (lo, hi) in zip(bucket_names, buckets):
            mask = (sibling_sizes >= lo) & (sibling_sizes <= hi)
            bucket_masks[name] = mask
            n_in_bucket = int(mask.sum())
            bucket_stats[name] = {
                'n_items': n_in_bucket,
                'fraction': float(n_in_bucket / N),
                'mean_size': float(sibling_sizes[mask].mean()) if n_in_bucket > 0 else 0.0,
                'median_size': float(np.median(sibling_sizes[mask])) if n_in_bucket > 0 else 0.0,
            }
            print(f'    桶 {name}: N={n_in_bucket}, fraction={bucket_stats[name]["fraction"]:.3f}')

        # Compute intra-sibling codebook diversity (how many unique z[l] codes within each sibling set)
        # Take average per bucket
        diversity_stats = {}
        for name, mask in bucket_masks.items():
            indices = np.where(mask)[0]
            if len(indices) == 0:
                diversity_stats[name] = {'mean_unique_l1_codes': 0, 'mean_unique_l2_codes': 0}
                continue
            unique_l_per = []
            unique_lt_per = []
            sample_indices = indices[:200] if len(indices) > 200 else indices  # subsample for speed
            for i in sample_indices:
                siblings = get_siblings(Z, int(i), l)
                if l < 3:
                    sub_codes_l = Z[l, siblings].numpy()
                    unique_l_per.append(len(np.unique(sub_codes_l)))
                if l + 1 <= 2 and l + 1 < 3:
                    # also check next layer
                    pass
            diversity_stats[name] = {
                'mean_unique_l_per_sibling': float(np.mean(unique_l_per)) if unique_l_per else 0.0,
                'std': float(np.std(unique_l_per)) if unique_l_per else 0.0,
                'n_sampled_siblings': len(sample_indices),
            }
            print(f'    桶 {name}: sibling 内 unique l+1 codes 均值 = {diversity_stats[name].get("mean_unique_l_per_sibling", 0.0):.2f}')

        results[layer_name] = {
            'sibling_size_stats': {
                'min': int(sibling_sizes.min()),
                'max': int(sibling_sizes.max()),
                'mean': float(sibling_sizes.mean()),
                'median': float(np.median(sibling_sizes)),
                'std': float(np.std(sibling_sizes)),
            },
            'buckets': bucket_stats,
            'diversity': diversity_stats,
            'note': 'L1 是空 prefix, 所有 item 单一大 sibling set',
        }

    # Save
    out = {
        'n_items': N,
        'sid_tensor_path': SID_TENSOR,
        'note': 'structural analysis only. TIGER model rank improvement (ΔRank) 需要 GPU 重跑 (见 task45_full).',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'sibling_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/sibling_metrics.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 303 Verdict: 精确 Sibling Ranking (结构分析)\n\n')
        f.write(f'数据集: Toys (N={N} items)\n\n')
        f.write('## 各层 sibling set 大小\n\n')
        f.write('| Layer | min | max | mean | median | std |\n')
        f.write('|-------|-----|-----|------|--------|-----|\n')
        for l_name, r in results.items():
            s = r['sibling_size_stats']
            f.write(f'| {l_name} | {s["min"]} | {s["max"]} | {s["mean"]:.2f} | {s["median"]:.1f} | {s["std"]:.2f} |\n')
        f.write('\n## 桶分布\n\n')
        for l_name, r in results.items():
            f.write(f'### {l_name}\n\n')
            f.write('| Bucket | N items | Fraction | Mean Size |\n')
            f.write('|--------|---------|----------|-----------|\n')
            for b_name in bucket_names:
                bs = r['buckets'][b_name]
                f.write(f'| {b_name} | {bs["n_items"]} | {bs["fraction"]:.3f} | {bs["mean_size"]:.2f} |\n')
            f.write('\n')
            f.write('**Sibling 内 z[l+1] code diversity (avg per sibling set)**:\n\n')
            for b_name in bucket_names:
                ds = r['diversity'][b_name]
                if 'mean_unique_l_per_sibling' in ds:
                    f.write(f'- {b_name}: 平均 {ds["mean_unique_l_per_sibling"]:.2f} ± {ds["std"]:.2f} 个 unique codes, '
                            f'采样 {ds["n_sampled_siblings"]} sibling sets\n')
                else:
                    f.write(f'- {b_name}: l=0 空 prefix 不区分\n')
            f.write('\n')

        f.write('## 判读\n\n')
        for l_name, r in results.items():
            s = r['sibling_size_stats']
            l_idx = int(l_name[1]) - 1
            f.write(f'### {l_name} (l={l_idx})\n')
            if l_idx == 0:
                f.write(f'- 所有 item 形成单一 sibling set (size = {N})\n')
                f.write(f'- 浅层无 prefix 区分能力\n\n')
                continue
            f.write(f'- sibling size mean={s["mean"]:.2f}, median={s["median"]:.1f} → '
                    f'prefix partition "size vs count" trade-off\n')
            # Find biggest bucket fraction
            biggest_bucket = max(r['buckets'].items(), key=lambda x: x[1]['fraction'])
            if biggest_bucket[1]['fraction'] > 0.5:
                f.write(f'- majority items ({biggest_bucket[1]["fraction"]:.1%}) are in bucket {biggest_bucket[0]} → '
                        f'sibling sets are mostly small but numerous\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()
