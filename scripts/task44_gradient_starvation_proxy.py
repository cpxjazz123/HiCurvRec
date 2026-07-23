#!/usr/bin/env python3
"""Task 302: 真实梯度饥饿诊断 (proxy 版, CPU only)

由于没有训练时 hook 数据, 用最终模型状态做结构性 proxy:
- A_l: active code ratio (1 - dead_code_ratio)
- D_l: dead code ratio
- effective_code_utilization: 考虑每 code 实际承载 item 数
- item_churn_proxy: 利用 prefix 重复度估计 churn 难易

完整版需训练时挂 hook 记录 G_l, U_l, churn 曲线.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task44_gradient_starvation'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def main():
    print('=' * 70)
    print('Task 302: 真实梯度饥饿诊断 (proxy 版, 最终状态)')
    print('=' * 70)

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    r_lst = bundle['r_lst']

    N = idx_lst[0].shape[0]
    print(f'\n[1] 加载 bundle, N={N}, L={len(codebooks)}')

    results = {}
    for l in range(len(codebooks)):
        K_l = codebooks[l].shape[0]
        idx_l = idx_lst[l].numpy().astype(int)
        counts = np.bincount(idx_l, minlength=K_l)
        active_codes = counts > 0
        n_active = int(active_codes.sum())
        A_l = float(n_active / K_l)
        D_l = 1.0 - A_l

        # Mean items per active code
        items_per_active = counts[active_codes]
        mean_items_per_active = float(items_per_active.mean()) if n_active > 0 else 0
        max_items_per_code = int(counts.max())
        min_items_per_code = int(counts[active_codes].min()) if n_active > 0 else 0

        # Residual norm at this layer (final state)
        r_l = r_lst[l].float().norm(dim=-1).numpy()
        mean_r_norm = float(r_l.mean())
        median_r_norm = float(np.median(r_l))

        # Codebook norm distribution (as proxy for "is the codebook still being updated?")
        cb_norm = codebooks[l].float().norm(dim=-1).numpy()
        cb_norm_mean = float(cb_norm.mean())
        cb_norm_std = float(cb_norm.std())
        cb_norm_max = float(cb_norm.max())
        cb_norm_min = float(cb_norm.min())

        # Codebook utilization Gini
        if n_active > 1:
            sorted_counts = np.sort(counts[active_codes])
            n_act = len(sorted_counts)
            cum = np.cumsum(sorted_counts)
            gini = float((2 * np.sum((np.arange(1, n_act + 1)) * sorted_counts)) / (n_act * cum[-1]) - (n_act + 1) / n_act)
        else:
            gini = 0.0

        results[f'L{l+1}'] = {
            'K': K_l,
            'n_active': n_active,
            'A_l_active_ratio': A_l,
            'D_l_dead_ratio': D_l,
            'mean_items_per_active': mean_items_per_active,
            'max_items_per_code': max_items_per_code,
            'min_items_per_code': min_items_per_code,
            'residual_norm_mean': mean_r_norm,
            'residual_norm_median': median_r_norm,
            'codebook_norm_mean': cb_norm_mean,
            'codebook_norm_std': cb_norm_std,
            'codebook_norm_max': cb_norm_max,
            'codebook_norm_min': cb_norm_min,
            'utilization_gini': gini,
        }
        print(f'\n  L{l+1} (K={K_l}):')
        print(f'    A_l = {A_l:.3f}, D_l = {D_l:.3f}')
        print(f'    items/code: mean={mean_items_per_active:.1f}, max={max_items_per_code}, min={min_items_per_code}')
        print(f'    residual norm: mean={mean_r_norm:.2f}, median={median_r_norm:.2f}')
        print(f'    codebook norm: mean={cb_norm_mean:.2f}, std={cb_norm_std:.2f}')
        print(f'    utilization Gini = {gini:.3f}')

    # Cross-layer comparison: gradient starvation proxy (GR_l)
    # proxy GR_l = (r_norm_mean / cb_norm_mean) ratio vs L1
    base = results['L1']['residual_norm_mean'] / (results['L1']['codebook_norm_mean'] + 1e-8)
    gr_proxy = {}
    for l in range(len(codebooks)):
        layer_name = f'L{l+1}'
        r = results[layer_name]
        ratio = r['residual_norm_mean'] / (r['codebook_norm_mean'] + 1e-8)
        gr = float(ratio / base)
        gr_proxy[layer_name] = gr
        results[layer_name]['GR_l_proxy'] = gr
        print(f'\n  {layer_name} GR_l proxy = {gr:.3f}')

    # Save
    out = {
        'n_items': N,
        'rqidx_path': RQIDX,
        'note': 'Proxy: 用最终模型状态计算 A_l, D_l, residual/codebook norm, utilization Gini. 完整版需训练时 hook 记录 G_l(t), U_l(t), churn.',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'gradient_starvation_proxy.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/gradient_starvation_proxy.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 302 Verdict: 真实梯度饥饿诊断 (proxy 版)\n\n')
        f.write(f'数据集: Toys (N={N}), 结构性 proxy 评估\n\n')
        f.write('## 各层活性 & 死码\n\n')
        f.write('| Layer | K | Active | A_l | D_l | items/code (mean/max/min) | Util Gini |\n')
        f.write('|-------|---|--------|-----|-----|--------------------------|-----------|\n')
        for l in range(len(codebooks)):
            r = results[f'L{l+1}']
            f.write(f'| L{l+1} | {r["K"]} | {r["n_active"]} | {r["A_l_active_ratio"]:.3f} | '
                    f'{r["D_l_dead_ratio"]:.3f} | {r["mean_items_per_active"]:.1f} / '
                    f'{r["max_items_per_code"]} / {r["min_items_per_code"]} | {r["utilization_gini"]:.3f} |\n')
        f.write('\n## Residual & Codebook Norm\n\n')
        f.write('| Layer | r_norm mean | r_norm median | cb_norm mean | cb_norm std | cb_norm min/max |\n')
        f.write('|-------|-------------|---------------|--------------|-------------|------------------|\n')
        for l in range(len(codebooks)):
            r = results[f'L{l+1}']
            f.write(f'| L{l+1} | {r["residual_norm_mean"]:.2f} | {r["residual_norm_median"]:.2f} | '
                    f'{r["codebook_norm_mean"]:.2f} | {r["codebook_norm_std"]:.2f} | '
                    f'{r["codebook_norm_min"]:.2f} / {r["codebook_norm_max"]:.2f} |\n')
        f.write('\n## GR_l Proxy\n\n')
        f.write('GR_l_proxy = (r_norm / cb_norm)_l / (r_norm / cb_norm)_L1\n\n')
        f.write('| Layer | GR_l_proxy |\n')
        f.write('|-------|------------|\n')
        for l in range(len(codebooks)):
            layer_name = f'L{l+1}'
            f.write(f'| {layer_name} | {gr_proxy[layer_name]:.3f} |\n')
        f.write('\n## 判读\n\n')
        for l in range(len(codebooks)):
            r = results[f'L{l+1}']
            f.write(f'### L{l+1}\n')
            if r['A_l_active_ratio'] < 0.7:
                f.write(f'- ⚠️ A_l = {r["A_l_active_ratio"]:.3f} < 0.7 → **大量 dead code**, 高度饥饿嫌疑\n')
            elif r['A_l_active_ratio'] < 0.85:
                f.write(f'- A_l = {r["A_l_active_ratio"]:.3f} < 0.85 → 一定饥饿嫌疑\n')
            else:
                f.write(f'- A_l = {r["A_l_active_ratio"]:.3f} ≥ 0.85 → 激活率充分\n')
            if r['utilization_gini'] > 0.4:
                f.write(f'- ⚠️ Gini = {r["utilization_gini"]:.3f} > 0.4 → **分布极不均衡**, 部分 code 过载\n')
            if r['codebook_norm_std'] < 0.5:
                f.write(f'- ⚠️ codebook norm std = {r["codebook_norm_std"]:.2f} 极小 → codebook 更新可能停滞\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()