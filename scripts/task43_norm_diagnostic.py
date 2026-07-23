#!/usr/bin/env python3
"""Task #43 — fused_64d / 3 subspace norm 分布诊断 (训练稳定性前置检查).

问题: mean=0.79, max=381 → 500× 极端长尾。未归一化的、方差巨大的输入直接喂 RQ-VAE 有风险:
  - 距离计算被极端 norm item 主导
  - codebook 往离群点方向坍缩
  - 离群 item 量化误差异常大,拖累整体 loss

输出:
  - norm 直方图 (4 张: fused, subspace_0/1/2)
  - percentile 统计 (p50/p90/p99/p99.9/max)
  - "是否需要 normalize"的建议
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec')
MCKG_EMB_PATH = ROOT / 'products/task99_mckg_rebuild/entity_embedding.pt'
OUT_DIR = ROOT / 'products/task43_norm_diag'
OUT_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print('⚠️ matplotlib 未装, 只输出统计数字')


def norm_stats(arr, name):
    norms = np.linalg.norm(arr, axis=1)
    pcts = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 99.95, 100]
    pct_vals = np.percentile(norms, pcts)
    stats = {
        'name': name,
        'shape': list(arr.shape),
        'count': int(len(norms)),
        'min': float(norms.min()),
        'max': float(norms.max()),
        'mean': float(norms.mean()),
        'std': float(norms.std()),
        'p10': float(pct_vals[0]),
        'p25': float(pct_vals[1]),
        'p50': float(pct_vals[2]),
        'p75': float(pct_vals[3]),
        'p90': float(pct_vals[4]),
        'p95': float(pct_vals[5]),
        'p99': float(pct_vals[6]),
        'p99.5': float(pct_vals[7]),
        'p99.9': float(pct_vals[8]),
        'p99.95': float(pct_vals[9]),
        'p100': float(pct_vals[10]),
        'n_above_10': int((norms > 10).sum()),
        'n_above_50': int((norms > 50).sum()),
        'n_above_100': int((norms > 100).sum()),
        'max_to_mean_ratio': float(norms.max() / norms.mean()),
        'p99_to_median_ratio': float(pct_vals[6] / pct_vals[2]),
    }
    return norms, stats


def main():
    print('========== Task #43 fused_64d / 3 subspace norm 长尾诊断 ==========')
    mckg = torch.load(MCKG_EMB_PATH, weights_only=False)
    sub_item = mckg['subspace_item'].numpy()  # (3, 11924, 64)
    fused = mckg['fused_item'].numpy()  # (11924, 64)

    spaces = [
        ('subspace_0_sphere', sub_item[0]),
        ('subspace_1_euclid', sub_item[1]),
        ('subspace_2_hyperbolic', sub_item[2]),
        ('fused_64d', fused),
    ]

    all_stats = []
    all_norms = {}
    for name, arr in spaces:
        norms, stats = norm_stats(arr, name)
        all_stats.append(stats)
        all_norms[name] = norms
        print(f'\n--- {name} ---')
        for k, v in stats.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')

    # Histograms
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (name, norms) in zip(axes.flatten(), all_norms.items()):
            # clip 到 p100 让图清晰
            log_norms = np.log10(norms.clip(min=1e-6))
            ax.hist(log_norms, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
            ax.axvline(np.log10(norms.mean()), color='red', linestyle='--', label=f'mean={norms.mean():.2f}')
            ax.axvline(np.log10(np.percentile(norms, 99)), color='orange', linestyle='--', label=f'p99={np.percentile(norms, 99):.2f}')
            ax.axvline(np.log10(norms.max()), color='darkred', linestyle='--', label=f'max={norms.max():.2f}')
            ax.set_xlabel('log10(L2 norm)')
            ax.set_ylabel('count')
            ax.set_title(f'{name} (n={len(norms)}, shape={dict(name="",shape="")})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle('Task #43 — Norm 分布 (log scale) — fused_64d / 3 subspace', fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'norm_histograms.png', dpi=100)
        print(f'\n直方图: {OUT_DIR / "norm_histograms.png"}')

    # 决策建议
    print('\n========== 决策建议 ==========')
    for stats in all_stats:
        name = stats['name']
        max_mean = stats['max_to_mean_ratio']
        p99_med = stats['p99_to_median_ratio']
        n_above_50 = stats['n_above_50']
        n_total = stats['count']

        verdict = '✅ 可直接喂 RQ-VAE (norm 健康)'
        action = 'no_norm'
        reason = f'max/mean={max_mean:.1f}×, p99/median={p99_med:.1f}×'

        if max_mean > 100 or p99_med > 50:
            verdict = '🔴 长尾极端 — 必须 clip / normalize'
            action = 'must_normalize'
            reason = f'max/mean={max_mean:.1f}× > 100×, p99/median={p99_med:.1f}× > 50×'
        elif max_mean > 20 or p99_med > 10:
            verdict = '🟡 长尾较重 — 建议 L2 normalize'
            action = 'recommend_normalize'
            reason = f'max/mean={max_mean:.1f}× > 20×, p99/median={p99_med:.1f}× > 10×'
        elif max_mean > 5 or p99_med > 5:
            verdict = '🟢 轻微长尾 — 可选 normalize'
            action = 'optional_normalize'
            reason = f'max/mean={max_mean:.1f}×, p99/median={p99_med:.1f}×'

        # 检查 n_above_50 是否占大头
        if n_above_50 > n_total * 0.05:  # > 5% items 离群
            verdict += f' [⚠️ {n_above_50} items ({100*n_above_50/n_total:.1f}%) norm>50]'
            action = 'must_clip'

        stats['verdict'] = verdict
        stats['action'] = action
        stats['reason'] = reason
        print(f'{name:30s}  {verdict}')
        print(f'  原因: {reason}')

    # 写 summary JSON
    summary = {
        'task': 'Task #43 — norm 长尾诊断 (训练稳定性前置检查)',
        'input': str(MCKG_EMB_PATH),
        'spaces': all_stats,
        'recommendation': {
            'fused_64d_for_task40': next(s['action'] for s in all_stats if s['name'] == 'fused_64d'),
            'subspace_for_task41': [s['action'] for s in all_stats if s['name'].startswith('subspace_')],
        },
    }
    out = OUT_DIR / 'task43_norm_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n输出: {out}')

    # 给出对 Task #40/#41 的具体 norm 处理建议
    print('\n========== Task #40/#41 norm 处理建议 ==========')
    fused_stats = next(s for s in all_stats if s['name'] == 'fused_64d')
    if fused_stats['action'] in ('must_normalize', 'must_clip', 'recommend_normalize'):
        print(f'⚠️ fused_64d {fused_stats["action"]}: max/mean={fused_stats["max_to_mean_ratio"]:.1f}×')
        print(f'   建议: 在 Stage 2.1 训练前 L2 normalize fused_64d')
        print(f'        fused_norm = fused / (||fused||_2 / target_norm)')
        print(f'        其中 target_norm = p95 norm = {fused_stats["p95"]:.4f}')

    sub_stats = [s for s in all_stats if s['name'].startswith('subspace_')]
    if any(s['action'] in ('must_normalize', 'must_clip') for s in sub_stats):
        print(f'\n⚠️ 部分 subspace 长尾较重 (max/mean > 100×):')
        for s in sub_stats:
            if s['action'] in ('must_normalize', 'must_clip'):
                print(f'   {s["name"]}: max/mean={s["max_to_mean_ratio"]:.1f}× — 建议 clip norm 到 p99={s["p99"]:.2f}')


if __name__ == '__main__':
    main()
