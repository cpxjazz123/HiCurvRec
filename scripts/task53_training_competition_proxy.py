#!/usr/bin/env python3
"""Task 311: 训练竞争诊断 (lightweight structural proxy, 不重训 12+12)

不重训 12 RQ-VAE + 12 TIGER, 而用结构性指标评估 4 种训练方法的潜力:
1. baseline RQ-VAE (已有)
2. loss normalization: 当前每层 loss 实际贡献是否均衡
3. shallow stop-grad: 浅层梯度贡献 (residual norm × codebook norm)
4. depth-balanced loss: 是否需要人为权重

指标:
- DRI_l = ΔUtility_l / |ΔReconstruction_l|
- 4 方法各自的潜在 gain 上限
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task53_training_competition'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def main():
    print('=' * 70)
    print('Task 311: 训练竞争诊断 (lightweight structural proxy)')
    print('=' * 70)

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    r_lst = bundle['r_lst']
    q_lst = bundle.get('q_lst', None)
    N = idx_lst[0].shape[0]

    # Per-layer metrics
    print('\n[1] 各层结构指标...')
    layer_stats = []
    for l in range(3):
        r_l = r_lst[l].float()  # input to layer l
        cb_l = codebooks[l].float()
        idx_l = idx_lst[l]
        q_l = cb_l[idx_l].float()
        r_after = (r_l - q_l).float()  # output residual after layer l

        # Reconstruction contribution at this layer
        recon_before = r_l.norm(dim=-1).mean().item()
        recon_after = r_after.norm(dim=-1).mean().item()
        recon_contrib = recon_before - recon_after  # how much layer l "explains"

        # Codebook diversity
        cb_norm = cb_l.norm(dim=-1).numpy()
        cb_norm_std = float(cb_norm.std())
        cb_norm_mean = float(cb_norm.mean())

        # Codebook member utilization
        counts = np.bincount(idx_l.numpy(), minlength=cb_l.shape[0])
        active = (counts > 0).sum()
        active_ratio = float(active / cb_l.shape[0])

        # Output residual norm (after layer l) - input to layer l+1
        out_norm = r_after.norm(dim=-1).mean().item()

        layer_stats.append({
            'layer': l + 1,
            'recon_before': recon_before,
            'recon_after': recon_after,
            'recon_contrib': recon_contrib,
            'cb_norm_mean': cb_norm_mean,
            'cb_norm_std': cb_norm_std,
            'active_ratio': active_ratio,
            'out_norm': out_norm,
        })
        print(f'  L{l+1}: recon contrib={recon_contrib:.3f}, cb_norm={cb_norm_mean:.3f}±{cb_norm_std:.3f}, '
              f'active={active_ratio:.3f}')

    # Method 1: baseline RQ-VAE
    print('\n[2] Method 1: Baseline...')
    m1_total_recon = sum(ls['recon_contrib'] for ls in layer_stats)
    m1 = {
        'method': 'baseline_RQ_VAE',
        'total_recon': m1_total_recon,
        'recon_per_layer': [ls['recon_contrib'] for ls in layer_stats],
        'note': '当前 baseline',
    }
    print(f'  total recon contrib: {m1_total_recon:.3f}')

    # Method 2: loss normalization (proportional to residual norm)
    # 如果按 ||r_l|| 倒数加权 loss:
    # - 当前 loss = ||r_after||², 即让每层均匀减小残差
    # - normal: 让每层 contribution 比例 ≈ ||r_l||
    print('\n[3] Method 2: Loss Normalization...')
    m2_weights = [ls['recon_before'] for ls in layer_stats]
    m2_weights = np.array(m2_weights) / np.sum(m2_weights)
    # 假设 normalized 后每层贡献更均衡
    m2_implied_recon = []
    for l in range(3):
        # 假设 normalized 后每层贡献与 r_before 成正比
        ideal_contrib = m2_weights[l] * m1_total_recon
        m2_implied_recon.append(ideal_contrib)
    m2_potential_gain = max(m2_implied_recon) - max(ls['recon_contrib'] for ls in layer_stats)
    m2 = {
        'method': 'loss_normalization',
        'weights': m2_weights.tolist(),
        'implied_recon': m2_implied_recon,
        'potential_gain': float(m2_potential_gain),
        'note': 'loss_normalization 潜在收益',
    }
    print(f'  weights: {m2_weights}')
    print(f'  potential gain: {m2_potential_gain:.3f}')

    # Method 3: shallow stop-grad (浅层不更新, 只更新深层)
    # 当前浅层 norm 大, 深层 norm 小, 浅层可能"耗尽"梯度
    # shallow stop-grad 假设: 让浅层 frozen, 梯度全给深层
    print('\n[4] Method 3: Shallow Stop-Grad...')
    # 估计: 如果浅层 L1 frozen, 深层 L2/L3 可获得更多 gradient
    # 假设浅层贡献 0, 深层按现有比例放大
    deep_total = sum(ls['recon_contrib'] for l, ls in enumerate(layer_stats) if l >= 1)
    shallow_contrib = layer_stats[0]['recon_contrib']
    m3_implied = layer_stats[0]['recon_contrib']  # L1 same
    m3_deep_avg = deep_total / 2
    m3_deep_orig_avg = (layer_stats[1]['recon_contrib'] + layer_stats[2]['recon_contrib']) / 2
    m3_potential = (m3_deep_avg - m3_deep_orig_avg) * 2  # 2 deep layers
    m3 = {
        'method': 'shallow_stop_grad',
        'shallow_contrib': shallow_contrib,
        'deep_total_orig': deep_total,
        'deep_total_implied': m3_deep_avg * 2,
        'potential_gain': float(m3_potential),
        'note': 'shallow stop-grad 假设浅层贡献冻结, 深层获得更多 gradient',
    }
    print(f'  shallow contrib: {shallow_contrib:.3f}')
    print(f'  deep total: orig={deep_total:.3f}, implied={m3_deep_avg*2:.3f}')
    print(f'  potential gain: {m3_potential:.3f}')

    # Method 4: depth-balanced loss (人为加大深层权重)
    print('\n[5] Method 4: Depth-Balanced Loss...')
    # 当前 L3 recon_contrib vs L1 比例
    ratio = layer_stats[2]['recon_contrib'] / (layer_stats[0]['recon_contrib'] + 1e-6)
    # 假设 balanced: 让 L3 contrib = L1 contrib
    target_l3 = layer_stats[0]['recon_contrib']
    m4_l3_gain = target_l3 - layer_stats[2]['recon_contrib']
    m4 = {
        'method': 'depth_balanced_loss',
        'orig_L3_recon': layer_stats[2]['recon_contrib'],
        'target_L3_recon': target_l3,
        'L3_implied_gain': float(m4_l3_gain),
        'ratio_L3_L1_orig': float(ratio),
        'note': '深度平衡 loss: 让 L3 contribution = L1',
    }
    print(f'  orig L3/L1 ratio: {ratio:.3f}')
    print(f'  implied L3 gain: {m4_l3_gain:.3f}')

    # DRI_l (depth-reconstruction-investment ratio)
    print('\n[6] DRI_l = recon_contrib / cb_norm²...')
    dri = []
    for ls in layer_stats:
        dri_l = ls['recon_contrib'] / (ls['cb_norm_mean'] ** 2 + 1e-6)
        dri.append(dri_l)
        print(f'  L{ls["layer"]}: DRI = {dri_l:.3f}')

    # Save
    out = {
        'n_items': N,
        'rqidx_path': RQIDX,
        'note': 'structural proxy: 不重训, 用现有 RQ-VAE 残差/codebook 估计 4 训法潜力. 完整版需训 12 RQ-VAE + 12 TIGER.',
        'layer_stats': layer_stats,
        'method_1_baseline': m1,
        'method_2_loss_norm': m2,
        'method_3_shallow_stop_grad': m3,
        'method_4_depth_balanced': m4,
        'DRI_per_layer': dri,
    }
    with open(os.path.join(OUT_DIR, '4methods_per_layer.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/4methods_per_layer.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 311 Verdict: 训练竞争诊断 (structural proxy)\n\n')
        f.write(f'数据集: Toys (N={N})\n\n')
        f.write('## 各层基础指标\n\n')
        f.write('| Layer | recon_before | recon_after | recon_contrib | cb_norm | active |\n')
        f.write('|-------|--------------|-------------|---------------|---------|--------|\n')
        for ls in layer_stats:
            f.write(f'| L{ls["layer"]} | {ls["recon_before"]:.3f} | {ls["recon_after"]:.3f} | '
                    f'{ls["recon_contrib"]:.3f} | {ls["cb_norm_mean"]:.3f} | {ls["active_ratio"]:.3f} |\n')
        f.write('\n## 4 方法潜在 gain\n\n')
        f.write('| Method | Potential Gain | Note |\n')
        f.write('|--------|----------------|------|\n')
        f.write(f'| baseline | 0 (已实现) | 当前状态 |\n')
        f.write(f'| loss_normalization | {m2_potential_gain:.3f} | 按 r_before 加权 |\n')
        f.write(f'| shallow_stop_grad | {m3_potential:.3f} | 浅层 frozen |\n')
        f.write(f'| depth_balanced_loss | {m4_l3_gain:.3f} | L3 权重加大 |\n')
        f.write('\n## DRI_l (depth-reconstruction-investment)\n\n')
        f.write('| Layer | DRI_l |\n')
        f.write('|-------|-------|\n')
        for l, d in enumerate(dri):
            f.write(f'| L{l+1} | {d:.3f} |\n')
        f.write('\n## 判读\n\n')
        for ls in layer_stats:
            f.write(f'### L{ls["layer"]}\n')
            f.write(f'- recon_contrib = {ls["recon_contrib"]:.3f}, cb_norm = {ls["cb_norm_mean"]:.3f}, active = {ls["active_ratio"]:.3f}\n')
            if ls['recon_contrib'] < 0.1:
                f.write(f'  → 该层 reconstruction 贡献小, 可能需 loss reweighting\n')
            if ls['cb_norm_std'] < 0.05:
                f.write(f'  → codebook norm std 极小 → 该层可能已停止有效更新\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()