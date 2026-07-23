#!/usr/bin/env python3
"""Idea 4 DPP 现象 1: 归一化 Gram log-volume V vs 碰撞率 Spearman 相关

测什么:
- 对每套 checkpoint 每层码本, 算 V_l = log det(C̃_l^T C̃_l + εI)
  其中 C̃_l = 单位化码字矩阵 (M_l × d), M_l = 2^W_l
- 总体积 V = Σ_l V_l
- 对 ≥ 4 算法 (A/B/C/WF) 画 V vs collision_rate 散点图
- 算 Spearman ρ

要看到的现象:
- 强负相关 (Spearman ρ < -0.6, 体积小 → 碰撞高)
- GSRQ (碰撞 0.65) 应该是体积最小的之一

Kill 线:
- |ρ| < 0.3 或正相关 → 码字扎堆不是碰撞成因, 判死 DPP 正则方法线

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt (A)
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt (B)
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt (C)
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt (WF)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from scipy.stats import spearmanr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea4_dpp'
os.makedirs(OUT_DIR, exist_ok=True)

# 4 算法路径
RQIDX_PATHS = {
    'A_RQ_VAE':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_MMQ':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_GSRQ':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_K256_64_16': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}

# 碰撞率 (3-layer full SID), from task15/idea1/task19
COLLISION_RATES = {
    'A_RQ_VAE':       0.2374,  # task15 A_baseline 3-layer
    'B_MMQ':          0.1611,  # task15 B_mmq 3-layer
    'C_GSRQ':         0.6511,  # task15 C_gsrq 3-layer
    'WF_K256_64_16':  0.4171,  # idea1_WF retrained
    # AQ 0.2371 (task19_aq_run_info), HRQ: 需从 SID 计算
}


def compute_V_per_layer(codebooks, eps=1e-3):
    """V_l = log det(C̃_l^T C̃_l + εI)

    C̃_l: (M_l, d) 单位化码字矩阵
    返回 V_l 标量
    """
    V_per_layer = []
    for l, C_l in enumerate(codebooks):
        if C_l is None:
            V_per_layer.append(None)
            continue
        # 单位化 (per-codeword, 避免范数影响)
        norms = C_l.norm(dim=1, keepdim=True).clamp(min=1e-8)
        C_norm = C_l / norms
        # Gram 矩阵
        G = C_norm.T @ C_norm  # (d, d)
        # 加 εI
        d = G.shape[0]
        G = G + eps * torch.eye(d)
        # log det (via slogdet for numerical stability)
        sign, logabsdet = torch.linalg.slogdet(G)
        if sign.item() <= 0:
            # 如果矩阵不正定, 用 sum of log(diag) 替代
            logabsdet = torch.log(torch.diag(G) + 1e-8).sum()
        V_per_layer.append(logabsdet.item())
    return V_per_layer


def main():
    print('=' * 70)
    print('Idea 4 DPP 现象 1: V vs collision_rate Spearman')
    print('=' * 70)

    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            print(f'[WARN] {algo_name}: rqidx 文件不存在, 跳过')
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        codebooks = data['codebooks']  # list of (M_l, d) tensors
        idx_lst = data['idx_lst']      # list of (N,) long tensors

        # 算 V per layer
        V_per_layer = compute_V_per_layer(codebooks)
        V_total = sum(v for v in V_per_layer if v is not None)

        # 碰撞率
        collision = COLLISION_RATES.get(algo_name, None)

        # K_per_layer
        K_per_layer = [c.shape[0] if c is not None else None for c in codebooks]

        # 利用率: unique count / K
        util_per_layer = []
        for l, idx in enumerate(idx_lst):
            n_unique = len(idx.unique())
            util = n_unique / K_per_layer[l] if K_per_layer[l] else 0
            util_per_layer.append(util)

        print(f'\n[{algo_name}]')
        print(f'  K_per_layer: {K_per_layer}')
        print(f'  V_per_layer: {[round(v, 2) if v else None for v in V_per_layer]}')
        print(f'  V_total: {round(V_total, 2)}')
        print(f'  util_per_layer: {[round(u, 3) for u in util_per_layer]}')
        print(f'  collision_rate: {collision}')

        results.append({
            'algorithm': algo_name,
            'K_per_layer': K_per_layer,
            'V_per_layer': V_per_layer,
            'V_total': V_total,
            'util_per_layer': util_per_layer,
            'collision_rate': collision,
        })

    # 算 Spearman
    valid = [r for r in results if r['collision_rate'] is not None and r['V_total'] is not None]
    if len(valid) >= 3:
        V_vals = np.array([r['V_total'] for r in valid])
        C_vals = np.array([r['collision_rate'] for r in valid])
        rho, pval = spearmanr(V_vals, C_vals)
        print(f'\n[Spearman] V_total vs collision_rate: ρ = {rho:.3f}, p = {pval:.3f}')
        print(f'  n = {len(valid)} algorithms: {[r["algorithm"] for r in valid]}')
    else:
        rho, pval = None, None
        print('\n[Spearman] 数据点不足 (<3), 无法计算')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea4_phen1_volume_vs_collision.json')
    with open(out_json, 'w') as f:
        json.dump({
            'per_algorithm': results,
            'spearman': {'rho': rho, 'p': pval, 'n': len(valid) if valid else 0},
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task9_dpp_codebook_volume.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if len(valid) >= 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            V_vals = np.array([r['V_total'] for r in valid])
            C_vals = np.array([r['collision_rate'] for r in valid])
            ax.scatter(V_vals, C_vals, s=120, c='steelblue', edgecolor='k', zorder=3)
            for r in valid:
                ax.annotate(r['algorithm'],
                            (r['V_total'], r['collision_rate']),
                            fontsize=9, xytext=(8, 8), textcoords='offset points')
            ax.set_xlabel('V_total = Σ_l log det(C̃_l^T C̃_l + εI)')
            ax.set_ylabel('collision_rate (3-layer SID)')
            ax.set_title(f'Idea 4 DPP: V vs collision\nSpearman ρ = {rho:.3f}, p = {pval:.3f}, n = {len(valid)}')
            ax.grid(True, alpha=0.3)
            # 加趋势线
            if len(valid) >= 2:
                z = np.polyfit(V_vals, C_vals, 1)
                xs = np.linspace(V_vals.min(), V_vals.max(), 100)
                ax.plot(xs, z[0]*xs + z[1], 'r--', alpha=0.5, label=f'linear fit')
                ax.legend()
            out_png = os.path.join(OUT_DIR, 'volume_vs_collision_scatter.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=120)
            print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] 画图失败: {e}')

    return results, rho, pval


if __name__ == '__main__':
    main()