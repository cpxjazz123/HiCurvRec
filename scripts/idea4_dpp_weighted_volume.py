#!/usr/bin/env python3
"""Idea 4 DPP 补测 C: 局部密度加权体积 V^weighted

测什么:
- 对每层码本, 算每个码字 c_j 的 Voronoi cell 落入的 item 数量 = density_j
- 加权码字: w_j ∝ density_j, 归一化
- V_l^weighted = log det(Σ w_j ĉ_j ĉ_j^T + εI)
- 对比 V_unweighted (原) vs V_weighted 看哪个与 collision 的相关性更符合假设

假设:
- 若 V_weighted 与 collision 呈强负相关, 而 V_unweighted 呈正相关 → 证明"原始 V 被密度混淆了", DPP 应该用 density-weighted

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/{A,B,C}_baseline_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt
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

RQIDX_PATHS = {
    'A_RQ_VAE':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_MMQ':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_GSRQ':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_K256_64_16': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}

COLLISION_RATES = {
    'A_RQ_VAE':       0.2374,
    'B_MMQ':          0.1611,
    'C_GSRQ':         0.6511,
    'WF_K256_64_16':  0.4171,
}

EPS = 1e-3


def compute_voronoi_density(idx_lst, codebooks):
    """对每层, 算每个码字的 Voronoi cell 落入 item 数 (= usage count)
    返回: list of (M_l,) per layer 的密度向量
    """
    densities = []
    for l, idx in enumerate(idx_lst):
        K = codebooks[l].shape[0]
        # 用 idx 的频次直接作为 density (这是该码字被分配的 item 数)
        counts = torch.bincount(idx, minlength=K).float()
        densities.append(counts.numpy())
    return densities


def compute_V_per_layer(codebooks, densities, weighted=False, eps=EPS):
    """V_l = log det(Σ w_j ĉ_j ĉ_j^T + εI)
    weighted=False: 用 1/M 等权 (原版)
    weighted=True: 用 w_j ∝ density_j 加权
    """
    V_per_layer = []
    for l, C_l in enumerate(codebooks):
        norms = C_l.norm(dim=1, keepdim=True).clamp(min=1e-8)
        C_norm = C_l / norms  # (M_l, d) 单位化码字

        if weighted:
            d_vec = densities[l]  # (M_l,)
            w = torch.tensor(d_vec / max(d_vec.sum(), 1.0), dtype=C_norm.dtype)  # 归一化
        else:
            M = C_norm.shape[0]
            w = torch.ones(M, dtype=C_norm.dtype) / M  # 等权

        # 加权代码矩阵 = (M_l, d), 加权 Σ_j w_j c_j c_j^T = C^T diag(w) C
        G = C_norm.T @ (w.unsqueeze(1) * C_norm)  # (d, d)
        d = G.shape[0]
        G = G + eps * torch.eye(d, dtype=G.dtype)

        sign, logabsdet = torch.linalg.slogdet(G)
        if sign.item() <= 0:
            logabsdet = torch.log(torch.diag(G) + 1e-8).sum()
        V_per_layer.append(logabsdet.item())
    return V_per_layer


def main():
    print('=' * 70)
    print('Idea 4 DPP 补测 C: 局部密度加权体积 V_weighted')
    print('=' * 70)

    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        codebooks = data['codebooks']
        idx_lst = data['idx_lst']

        # 计算密度 (Voronoi cell 大小 = 该码字被分配的 item 数)
        densities = compute_voronoi_density(idx_lst, codebooks)

        # 原始 V (等权)
        V_uniform = compute_V_per_layer(codebooks, densities, weighted=False)
        # 加权 V
        V_weighted = compute_V_per_layer(codebooks, densities, weighted=True)

        # 汇总
        V_unif_total = sum(v for v in V_uniform if v is not None)
        V_w_total = sum(v for v in V_weighted if v is not None)

        collision = COLLISION_RATES[algo_name]

        # 密度统计
        dens_norms = []
        for d_per_layer in densities:
            if d_per_layer.sum() > 0:
                p = d_per_layer / d_per_layer.sum()
                H = -(p[p > 0] * np.log(p[p > 0])).sum()
                dens_norms.append(H)
            else:
                dens_norms.append(0.0)

        print(f'\n[{algo_name}]')
        print(f'  V_uniform per layer = {[round(v, 2) for v in V_uniform]}')
        print(f'  V_weighted per layer = {[round(v, 2) for v in V_weighted]}')
        print(f'  V_total: uniform = {round(V_unif_total, 2)}, weighted = {round(V_w_total, 2)}')
        print(f'  collision = {collision}')

        results.append({
            'algorithm': algo_name,
            'V_uniform_per_layer': [float(v) for v in V_uniform],
            'V_uniform_total': float(V_unif_total),
            'V_weighted_per_layer': [float(v) for v in V_weighted],
            'V_weighted_total': float(V_w_total),
            'density_entropy_per_layer': [float(h) for h in dens_norms],
            'collision_rate': collision,
        })

    # Spearman 相关 (uniform vs weighted)
    valid = [r for r in results if r['collision_rate'] is not None]
    if len(valid) >= 3:
        V_unif = np.array([r['V_uniform_total'] for r in valid])
        V_w = np.array([r['V_weighted_total'] for r in valid])
        C = np.array([r['collision_rate'] for r in valid])

        rho_unif, p_unif = spearmanr(V_unif, C)
        rho_w, p_w = spearmanr(V_w, C)

        print(f'\n[Spearman] V_uniform vs collision: ρ = {rho_unif:.3f}, p = {p_unif:.3f}')
        print(f'[Spearman] V_weighted vs collision: ρ = {rho_w:.3f}, p = {p_w:.3f}')

        # 判定
        # 假设: V_weighted 应该呈负相关 (大体积+密度匹配 = 低碰撞)
        if rho_w < -0.3 and rho_unif >= 0:
            verdict = 'FLIPPED_TO_NEG: 加权体积与碰撞负相关, 验证猜想 → 这条线值得未来深挖'
        elif rho_w < rho_unif:
            verdict = f'IMPROVED: 加权让相关从 +{rho_unif:.2f} 降到 +{rho_w:.2f}, 部分验证猜想'
        elif abs(rho_w - rho_unif) < 0.1:
            verdict = 'UNCHANGED: 加权前后相关性几乎不变, 猜想不成立'
        else:
            verdict = f'OTHER: 相关性变化复杂 (unif={rho_unif:.2f}, weighted={rho_w:.2f}), 需进一步分析'
    else:
        verdict = 'INSUFFICIENT_DATA'

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea4_supplementary_testC.json')
    with open(out_json, 'w') as f:
        json.dump({
            'per_algorithm': results,
            'spearman': {
                'uniform': {'rho': float(rho_unif), 'p': float(p_unif)},
                'weighted': {'rho': float(rho_w), 'p': float(p_w)},
            },
            'verdict': verdict,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task9_dpp_codebook_volume.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            ax.scatter(r['V_uniform_total'], r['V_weighted_total'],
                       s=120, edgecolor='k', zorder=3)
            ax.annotate(r['algorithm'],
                        (r['V_uniform_total'], r['V_weighted_total']),
                        fontsize=9, xytext=(8, 8), textcoords='offset points')
        # y=x 参考线
        lims = [min(min(r['V_uniform_total'] for r in results),
                    min(r['V_weighted_total'] for r in results)) - 200,
                max(max(r['V_uniform_total'] for r in results),
                    max(r['V_weighted_total'] for r in results)) + 200]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('V_uniform_total')
        ax.set_ylabel('V_weighted_total')
        ax.set_title(f'Idea 4 DPP 补测 C: 等权 vs 加权体积\n'
                     f'ρ(V_uniform, C) = {rho_unif:.3f}\n'
                     f'ρ(V_weighted, C) = {rho_w:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(OUT_DIR, 'weighted_vs_uniform_scatter.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] 画图失败: {e}')

    return results, verdict


if __name__ == '__main__':
    main()
