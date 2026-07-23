#!/usr/bin/env python3
"""Task 300: 条件熵分支诊断 (H(Z_l), H(Z_l|Z_<l), B_eff, CR_l)

判断深层 SID 是真正提供新的离散分支 (B_eff 上升) 还是仅机械细分 (B_eff 接近 1).
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task42_entropy_branching'
os.makedirs(OUT_DIR, exist_ok=True)

SID_TENSOR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt'


def entropy_from_counts(counts):
    """Shannon entropy (natural log) from an array of counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def compute_layer_stats(Z, l):
    """Compute H(Z_l), H(Z_l|Z_<l), B_eff, CR_l_relative for layer l.

    Z has shape (L_total, N).
    Returns dict with all metrics.
    """
    N = Z.shape[1]
    if l == 0:
        # prefix: empty prefix, conditional = marginal
        # But task42 treats l=1, 2, 3 specifically. For l=0 in indexing (L1), prefix is empty
        counts = np.bincount(Z[0].numpy(), minlength=Z[0].max().item() + 1)
        h_l = entropy_from_counts(counts)
        h_cond = h_l  # empty prefix
        b_eff = float(np.exp(h_cond))
        novelty = 1.0  # = h_l / h_l
        # collision: pairs at l=0 (no prefix partition)
        # Use item-pair collisions based on same code at layer 0
        # C_0 = sum_c binom(|{i: z[i]=c}|, 2)
        c0 = sum(int(n*(n-1)/2) for n in counts if n >= 2)
        # C_l for l=0 same as C_0 since partition is the only one
        c_l = c0
        cr_l = 0.0
        return {
            'h_z_l': h_l,
            'h_z_l_given_z_lt_l': h_cond,
            'b_eff': b_eff,
            'b_eff_mean_per_prefix': b_eff,
            'novelty_l': novelty,
            'c_lt_l': c0,  # before adding layer l: C_{l-1} (with c_-1 = 0 here)
            'c_l': c_l,
            'cr_l': cr_l,
            'n_items': N,
            'n_unique_codes': int((counts > 0).sum()),
        }

    # l >= 1: partition by prefix Z[<l]
    Z_lt = Z[:l]  # shape (l, N)
    Z_l = Z[l]    # shape (N,)

    # group items by prefix
    prefix_keys = []
    for i in range(N):
        prefix_keys.append(tuple(int(Z_lt[k, i].item()) for k in range(l)))
    from collections import defaultdict
    groups = defaultdict(list)
    for i, key in enumerate(prefix_keys):
        groups[key].append(i)

    # Compute collisions BEFORE adding layer l: pairs sharing same prefix (l-1)
    c_before = 0
    for key, items in groups.items():
        n = len(items)
        if n >= 2:
            c_before += n*(n-1)//2

    # Compute collisions AFTER adding layer l: pairs sharing same (prefix, z_l) tuple
    after_groups = defaultdict(int)
    for key, items in groups.items():
        for i in items:
            full_key = key + (int(Z_l[i].item()),)
            after_groups[full_key] += 1
    c_after = sum(int(n*(n-1)/2) for n in after_groups.values() if n >= 2)

    # Compute H(Z_l) marginal
    counts_l = np.bincount(Z_l.numpy(), minlength=int(Z_l.max().item()) + 1)
    h_l = entropy_from_counts(counts_l)

    # Compute H(Z_l | Z_<l) conditional via grouped histograms
    h_cond_n = 0.0
    n_with_data = 0
    b_eff_total = 0.0
    b_eff_per_prefix = []
    for key, items in groups.items():
        if len(items) == 0:
            continue
        sub_counts = np.bincount(Z_l[items].numpy(), minlength=int(Z_l.max().item()) + 1)
        sub_h = entropy_from_counts(sub_counts)
        weight = len(items) / N
        h_cond_n += weight * sub_h
        b_eff_per_prefix.append(float(np.exp(sub_h)))
        n_with_data += 1

    # Average B_eff per prefix (for sanity)
    b_eff_mean_per_prefix = float(np.mean(b_eff_per_prefix)) if b_eff_per_prefix else 1.0

    # Effective branches: exp(average conditional entropy)
    b_eff = float(np.exp(h_cond_n))

    novelty = h_cond_n / (h_l + 1e-12)

    cr_l = 1.0 - c_after / (c_before + 1e-12)

    return {
        'h_z_l': h_l,
        'h_z_l_given_z_lt_l': h_cond_n,
        'b_eff': b_eff,
        'b_eff_mean_per_prefix': b_eff_mean_per_prefix,
        'novelty_l': float(novelty),
        'c_lt_l': c_before,
        'c_l': c_after,
        'cr_l': float(cr_l),
        'n_items': N,
        'n_unique_codes': int((counts_l > 0).sum()),
        'n_unique_prefixes': len(groups),
    }


def main():
    print('=' * 70)
    print('Task 300: 条件熵分支诊断 (H(Z_l), H(Z_l|Z_<l), B_eff, CR_l)')
    print('=' * 70)

    # Load SID tensor
    Z_full = torch.load(SID_TENSOR, map_location='cpu', weights_only=False)
    if Z_full.dim() == 2 and Z_full.shape[1] == 4:
        Z = Z_full.t().long()  # (4, N)
    elif Z_full.dim() == 2 and Z_full.shape[0] == 4:
        Z = Z_full.long()
    else:
        raise ValueError(f'Unexpected SID shape: {Z_full.shape}')
    N_total = Z.shape[1]
    print(f'\n[1] 加载 SID tensor: shape={tuple(Z.shape)}, N={N_total}')

    # Per the task definition, we test layers L1, L2, L3 (indices 0, 1, 2 in Z[:, 0..2])
    # Also include L4 (dedup digit, index 3) for completeness
    results = {}
    for l in range(4):
        layer_name = f'L{l+1}'
        print(f'\n[2.{l+1}] 计算 {layer_name} 统计...')
        stats = compute_layer_stats(Z, l)
        results[layer_name] = stats
        print(f'  H(Z_l)         = {stats["h_z_l"]:.4f}')
        print(f'  H(Z_l|Z_<l)    = {stats["h_z_l_given_z_lt_l"]:.4f}')
        print(f'  B_eff (全局)     = {stats["b_eff"]:.2f}')
        print(f'  B_eff (per pfx) = {stats["b_eff_mean_per_prefix"]:.2f}')
        print(f'  Novelty_l      = {stats["novelty_l"]:.4f}')
        print(f'  C_<l           = {stats["c_lt_l"]:,}')
        print(f'  C_l            = {stats["c_l"]:,}')
        print(f'  CR_l           = {stats["cr_l"]:.4f}')
        print(f'  unique codes   = {stats["n_unique_codes"]}')

    out = {
        'n_items': N_total,
        'sid_tensor_path': SID_TENSOR,
        'note': 'B_eff (全局) = exp(H(Z_l|Z_<l)) 平均 per item; B_eff (per pfx) = exp(H(Z_l|pfx)) 的组级平均',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'entropy_per_layer.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/entropy_per_layer.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 300 Verdict: 条件熵分支诊断\n\n')
        f.write(f'数据集: Toys (N={N_total})\n\n')
        f.write('## 各层统计\n\n')
        f.write('| Layer | H(Z_l) | H(Z_l\\|Z_<l) | B_eff | Novelty_l | CR_l | unique_codes |\n')
        f.write('|-------|--------|-------------|-------|-----------|------|--------------|\n')
        for l_name, stats in results.items():
            f.write(f'| {l_name} | {stats["h_z_l"]:.3f} | {stats["h_z_l_given_z_lt_l"]:.3f} | '
                    f'{stats["b_eff"]:.1f} | {stats["novelty_l"]:.3f} | {stats["cr_l"]:.4f} | '
                    f'{stats["n_unique_codes"]} |\n')
        f.write('\n## 解读\n\n')

        for l_name, stats in results.items():
            l_idx = int(l_name[1]) - 1
            b_eff = stats['b_eff']
            novelty = stats['novelty_l']
            cr = stats['cr_l']
            f.write(f'### {l_name} (index={l_idx})\n')
            f.write(f'- B_eff = {b_eff:.2f}: ')
            if b_eff > 50:
                f.write('大量有效分支\n')
            elif b_eff > 10:
                f.write('中等有效分支\n')
            else:
                f.write('分支数少（机械细分嫌疑）\n')
            f.write(f'- Novelty_l = {novelty:.3f}: ')
            if novelty > 0.9:
                f.write('条件熵接近边际熵 → 该层确实提供新信息\n')
            elif novelty > 0.5:
                f.write('中等条件独立\n')
            else:
                f.write('条件冗余 → 该层主要复制 prefix 信息\n')
            f.write(f'- CR_l = {cr:.4f}: ')
            if cr > 0.5:
                f.write('强 collision reduction（该层大量区分 item pair）\n')
            else:
                f.write('弱 collision reduction（该层区分能力不足）\n')
            f.write('\n')

        f.write('\n## 综合判读\n\n')
        # Check novelty trend
        novelties = [results[f'L{i+1}']['novelty_l'] for i in range(3)]
        b_effs = [results[f'L{i+1}']['b_eff'] for i in range(3)]
        crs = [results[f'L{i+1}']['cr_l'] for i in range(3)]

        if all(n > 0.85 for n in novelties):
            f.write('- 各层条件熵/边际熵均高 → 三层都提供显著新信息\n')
        if b_effs[2] > b_effs[0] * 2:
            f.write('- L3 的 B_eff 比 L1 高 → 深层确实增加有效分支\n')
        elif b_effs[2] < b_effs[0] * 1.5:
            f.write('- L3 的 B_eff 与 L1 相近 → 深层主要是重复 prefix\n')
        if crs[2] < 0.3:
            f.write('- L3 collision reduction 弱 → 该层区分 item pair 能力低\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()
