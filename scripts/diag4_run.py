#!/usr/bin/env python3
"""HRQ 诊断4: 流行度 vs Poincaré 中心距离 — 看 HRQ 输入空间是否自发地
   把冷门商品摆得离原点更远 (VarLenRec-like 隐式行为)

公式:
  d_hyp(o, h_i) 在 Poincaré ball 里:
    把 Lorentz h_i = [h_0, h_[1:]] 投到 Poincaré:
      p_i = h_[1:] / (1 + h_0)
    d_hyp(o, p_i) = 2 arctanh(||p_i||)
  Spearman correlation: ρ = Spearman(d_hyp(o, h_i), -log(pop_i + 1))

预期: 若 HRQ 隐式做了 VarLenRec 的事, 则 ρ 显著**负**(冷门→远).
kill 线: 没有相关 → 排除"流行度感知容量分配"机制.

复用:
- Stage 1 embedding (X_raw) → lift to Lorentz
- popularity from diag4_popularity_distance.toys_item_popularity.npy
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from scipy.stats import spearmanr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag4_pop_distance'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GenyRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt' if False else '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
POP_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag2_ollivier_ricci/toys_item_popularity.npy'

CURVATURE = 1.0  # Lorentz 曲率 (原 task20 设定)


def lift_to_lorentz(x, c=CURVATURE):
    x_norm2 = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)
    return torch.cat([h0, x], dim=-1)


def lorentz_to_poincare(h):
    """Lorentz h = [h_0, h_[1:]] → Poincaré p = h_[1:] / (h_0 + 1)."""
    h0 = h[..., 0:1]
    h1 = h[..., 1:]
    return h1 / (h0 + 1.0)


def poincare_distance_to_origin(p):
    """d_hyp(o, p) = 2 arctanh(||p||), for ||p|| < 1."""
    norms = p.norm(dim=-1)
    norms = torch.clamp(norms, max=1 - 1e-5)
    return 2.0 * torch.atanh(norms)


def main():
    print('=' * 70)
    print('HRQ 诊断4: 流行度 vs Poincaré 中心距离')
    print('=' * 70)

    print('\n[Step 1] 加载 embedding + popularity')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N, D = x_eu.shape
    print(f'  embedding: {tuple(x_eu.shape)}')

    pop = np.load(POP_PATH)
    print(f'  popularity: shape {pop.shape}, min/max/median: '
          f'{pop.min()}/{pop.max()}/{np.median(pop):.0f}')

    # 截断到 N (catalog size match)
    if len(pop) > N:
        pop = pop[:N]
    elif len(pop) < N:
        pop = np.concatenate([pop, np.zeros(N - len(pop), dtype=np.int32)])
    print(f'  popularity (truncated): {len(pop)} items')

    # 过滤零 popularity (log(0) 无意义)
    nonzero_mask = pop > 0
    print(f'  nonzero items: {nonzero_mask.sum()}/{N}')

    print('\n[Step 2] 加载 Lorentz 提升 → Poincaré ball')
    h = lift_to_lorentz(x_eu).float()  # (N, D+1)
    p = lorentz_to_poincare(h)         # (N, D)
    print(f'  Poincaré ball ||p|| stats: '
          f'min={p.norm(dim=-1).min():.4f}, '
          f'max={p.norm(dim=-1).max():.4f}, '
          f'median={p.norm(dim=-1).median():.4f}')

    print('\n[Step 3] 算 d_hyp(o, p_i) for each item')
    d_hyp = poincare_distance_to_origin(p).numpy()  # (N,)
    print(f'  d_hyp stats: min={d_hyp.min():.4f}, max={d_hyp.max():.4f}, '
          f'median={np.median(d_hyp):.4f}')

    log_pop = np.log(pop + 1)  # +1 避免 log 0
    neg_log_pop = -log_pop  # 流行度越低, neg_log_pop 越大

    # 全部 item: Spearman 相关 (包括 0 popularity)
    rho_all, p_all = spearmanr(d_hyp, neg_log_pop)
    print(f'\n[Step 4a] Spearman(d_hyp, -log(pop)) ALL items: ρ={rho_all:.4f}, p={p_all:.2e}')

    # 只取 nonzero popularity
    rho_nz, p_nz = spearmanr(d_hyp[nonzero_mask], neg_log_pop[nonzero_mask])
    print(f'[Step 4b] Spearman(d_hyp, -log(pop)) NONZERO items: ρ={rho_nz:.4f}, p={p_nz:.2e}')

    # 按 popularity 分位: 冷门 (low) vs 热门 (high)
    print(f'\n[Step 5] 冷热门对比')
    q33 = np.percentile(pop[nonzero_mask], 33)
    q67 = np.percentile(pop[nonzero_mask], 67)
    cold_mask = pop <= q33
    warm_mask = (pop >= q33) & (pop <= q67)
    hot_mask = pop >= q67

    d_cold = d_hyp[cold_mask & nonzero_mask].mean()
    d_warm = d_hyp[warm_mask & nonzero_mask].mean()
    d_hot = d_hyp[hot_mask & nonzero_mask].mean()
    pop_cold = pop[cold_mask & nonzero_mask].mean()
    pop_warm = pop[warm_mask & nonzero_mask].mean()
    pop_hot = pop[hot_mask & nonzero_mask].mean()

    print(f'  Cold (pop ≤ {q33:.0f}): n={(cold_mask & nonzero_mask).sum()}, '
          f'mean_pop={pop_cold:.1f}, mean_d_hyp={d_cold:.4f}')
    print(f'  Warm ({q33:.0f} < pop < {q67:.0f}): n={(warm_mask & nonzero_mask).sum()}, '
          f'mean_pop={pop_warm:.1f}, mean_d_hyp={d_warm:.4f}')
    print(f'  Hot (pop ≥ {q67:.0f}): n={(hot_mask & nonzero_mask).sum()}, '
          f'mean_pop={pop_hot:.1f}, mean_d_hyp={d_hot:.4f}')

    # 判定
    # 若 HRQ 隐式做了 VarLenRec (冷→远), ρ 应显著负 (< -0.1)
    if rho_nz < -0.10 and p_nz < 0.001:
        verdict = (
            f'CONFIRMED_VARLENREC_PATTERN: ρ_Spearman(d_hyp, -log(pop)) = {rho_nz:.4f} '
            f'(p={p_nz:.2e}), 显著负相关. HRQ 输入空间自发地把冷门商品放在了离 Poincaré 中心更远的位置. '
            f'与 VarLenRec 的"流行度感知容量分配"机制吻合, '
            f'给 +37% R@10 提供了一个具体、可解释、有文献佐证的机制解释.'
        )
        outcome = 'confirmed'
    elif rho_nz > 0.10 and p_nz < 0.001:
        verdict = (
            f'REVERSE_VARLENREC: ρ = {rho_nz:.4f} 显著正(冷门反而近中心). '
            f'与 VarLenRec 假设相反, HRQ 没做"冷门放远"的事. '
            f'机制需另寻.'
        )
        outcome = 'reverse'
    else:
        verdict = (
            f'KILL_POPULARITY_PATTERN: ρ = {rho_nz:.4f} (p={p_nz:.2e}), 无显著 Spearman 相关. '
            f'排除"流行度感知容量分配"这个具体机制. '
            f'+37% R@10 收益来自其他机制, 与"冷门放更远"无关.'
        )
        outcome = 'kill_no_correlation'

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'pop_distance.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'poincare_distance_vs_popularity',
            'description': (
                'Lift to Lorentz, project to Poincaré ball, '
                'd_hyp(o, h_i) = 2 arctanh(||p_i||). '
                'Spearman correlation with -log(pop_i + 1).'
            ),
            'params': {
                'curvature': CURVATURE,
                'n_items': N,
            },
            'stats': {
                'poincare_norm_min': float(p.norm(dim=-1).min()),
                'poincare_norm_max': float(p.norm(dim=-1).max()),
                'poincare_norm_median': float(p.norm(dim=-1).median()),
                'd_hyp_min': float(d_hyp.min()),
                'd_hyp_max': float(d_hyp.max()),
                'd_hyp_median': float(np.median(d_hyp)),
            },
            'spearman_results': {
                'all_items': {'rho': float(rho_all), 'p': float(p_all)},
                'nonzero_items': {'rho': float(rho_nz), 'p': float(p_nz)},
            },
            'cold_warm_hot': {
                'cold_n': int((cold_mask & nonzero_mask).sum()),
                'cold_mean_pop': float(pop_cold),
                'cold_mean_d_hyp': float(d_cold),
                'warm_n': int((warm_mask & nonzero_mask).sum()),
                'warm_mean_pop': float(pop_warm),
                'warm_mean_d_hyp': float(d_warm),
                'hot_n': int((hot_mask & nonzero_mask).sum()),
                'hot_mean_pop': float(pop_hot),
                'hot_mean_d_hyp': float(d_hot),
            },
            'verdict': verdict,
            'outcome': outcome,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.scatter(neg_log_pop[nonzero_mask], d_hyp[nonzero_mask], s=3, alpha=0.3)
        ax.set_xlabel('-log(pop + 1)')
        ax.set_ylabel('d_hyp(o, p_i)')
        ax.set_title(f'Scatter: ρ = {rho_nz:.3f}')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Bin by popularity, plot mean d_hyp per bin
        pop_nz = pop[nonzero_mask]
        d_nz = d_hyp[nonzero_mask]
        bins = np.percentile(pop_nz, np.linspace(0, 100, 11))
        bin_indices = np.digitize(pop_nz, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        bin_mean_pop = np.zeros(len(bins) - 1)
        bin_mean_d = np.zeros(len(bins) - 1)
        for b in range(len(bins) - 1):
            mask_b = bin_indices == b
            if mask_b.any():
                bin_mean_pop[b] = pop_nz[mask_b].mean()
                bin_mean_d[b] = d_nz[mask_b].mean()
        ax.plot(bin_mean_pop, bin_mean_d, 'o-', color='steelblue')
        ax.set_xlabel('popularity (mean per decile)')
        ax.set_ylabel('mean d_hyp(o, p_i)')
        ax.set_xscale('log')
        ax.set_title('Decile means: cooler→larger d_hyp?')
        ax.grid(True, alpha=0.3)

        plt.suptitle('HRQ 诊断4: 流行度 vs Poincaré 中心距离')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, 'pop_distance_scatter.png')
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] plot failed: {e}')

    return verdict, d_hyp, rho_nz, rho_all


if __name__ == '__main__':
    main()
