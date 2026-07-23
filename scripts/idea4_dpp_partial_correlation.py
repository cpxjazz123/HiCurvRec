#!/usr/bin/env python3
"""Idea 4 DPP 现象 2: 控制 H_util 的 V vs collision 偏相关

测什么:
- 对每套算法, 算 per-layer 利用率熵 H_l^{util} = -Σ_j p_j log p_j
  其中 p_j = count(idx == j) / N
- 总体利用率熵 H_util_total = Σ_l H_l^{util}
- 算 ρ_partial(V, collision | H_util_total):
  在控制利用率后, V 与 collision 的偏相关系数

要看到的现象:
- |ρ_partial| > 0.4 → 体积是独立于利用率的信号, DPP 抓到了现有指标抓不到的东西

Kill 线:
- |ρ_partial| < 0.2 → V 与 H_util 完全共线, DPP 正则不提供新信号

复用:
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/{A,B,C}_baseline_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt
- /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/idea4_dpp_volume_vs_collision.py (复用 V)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

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


def compute_H_util(idx_lst):
    """per-layer 利用率熵 H_l = -Σ_j p_j log p_j, p_j = count(idx == j) / N
    """
    H_per_layer = []
    for l, idx in enumerate(idx_lst):
        N = len(idx)
        # 频次统计
        K = idx.max().item() + 1
        counts = torch.bincount(idx, minlength=K).float()
        p = counts / N
        # 过滤 0
        p = p[p > 0]
        H = -(p * p.log()).sum().item()
        H_per_layer.append(H)
    return H_per_layer


def partial_corr(x, y, z):
    """偏相关 ρ(x, y | z), z 是控制变量
    用残差法: x_res = x - α*z - β, y_res = y - γ*z - δ, 然后 pearson(x_res, y_res)
    """
    from numpy.polynomial import polynomial as P
    # 拟合 x = a*z + b
    z_arr = np.asarray(z, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # x ~ z
    zx_cov = np.cov(z_arr, x_arr)
    beta_x = zx_cov[0, 1] / zx_cov[0, 0]
    x_res = x_arr - beta_x * (z_arr - z_arr.mean())

    # y ~ z
    zy_cov = np.cov(z_arr, y_arr)
    beta_y = zy_cov[0, 1] / zy_cov[0, 0]
    y_res = y_arr - beta_y * (z_arr - z_arr.mean())

    rho_p, _ = pearsonr(x_res, y_res)
    return float(rho_p)


def main():
    print('=' * 70)
    print('Idea 4 DPP 现象 2: 控制 H_util 的偏相关 V vs collision')
    print('=' * 70)

    # 加载现象 1 的 V
    phen1_path = os.path.join(OUT_DIR, 'idea4_phen1_volume_vs_collision.json')
    if not os.path.exists(phen1_path):
        print(f'[ERROR] 现象 1 数据文件不存在: {phen1_path}, 请先跑现象 1')
        return None

    with open(phen1_path) as f:
        phen1 = json.load(f)
    V_by_algo = {r['algorithm']: r['V_total'] for r in phen1['per_algorithm']}

    # 算每算法的 H_util
    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            print(f'[WARN] {algo_name}: rqidx 不存在, 跳过')
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        idx_lst = data['idx_lst']
        H_per_layer = compute_H_util(idx_lst)
        H_util_total = sum(H_per_layer)
        V_total = V_by_algo.get(algo_name, None)
        collision = COLLISION_RATES.get(algo_name, None)

        print(f'\n[{algo_name}]')
        print(f'  H_per_layer: {[round(h, 3) for h in H_per_layer]}')
        print(f'  H_util_total: {round(H_util_total, 3)}')
        print(f'  V_total: {round(V_total, 2) if V_total else None}')
        print(f'  collision: {collision}')

        results.append({
            'algorithm': algo_name,
            'H_per_layer': H_per_layer,
            'H_util_total': H_util_total,
            'V_total': V_total,
            'collision_rate': collision,
        })

    # 算偏相关
    valid = [r for r in results if r['collision_rate'] is not None and r['V_total'] is not None]
    if len(valid) >= 3:
        V = np.array([r['V_total'] for r in valid])
        C = np.array([r['collision_rate'] for r in valid])
        H = np.array([r['H_util_total'] for r in valid])

        # 偏相关 (V, C | H)
        rho_partial = partial_corr(V, C, H)
        # 原始 Spearman
        rho_spearman, p_spearman = spearmanr(V, C)
        # 控制前后
        print(f'\n[偏相关] ρ(V, C | H_util) = {rho_partial:.3f}')
        print(f'[Spearman] ρ(V, C) = {rho_spearman:.3f}, p = {p_spearman:.3f}')
        print(f'  n = {len(valid)} algorithms: {[r["algorithm"] for r in valid]}')

        # Kill 判定
        if abs(rho_partial) < 0.2:
            verdict = 'KILLED'
            reason = f'|ρ_partial| = {abs(rho_partial):.3f} < 0.2 → V 与 H_util 完全共线, DPP 不提供新信号'
        elif abs(rho_partial) > 0.4:
            verdict = 'PASS'
            reason = f'|ρ_partial| = {abs(rho_partial):.3f} > 0.4 → 体积是独立信号'
        else:
            verdict = 'INCONCLUSIVE'
            reason = f'|ρ_partial| = {abs(rho_partial):.3f} 介于 0.2-0.4, 需要更多算法'
    else:
        rho_partial = None
        rho_spearman = None
        verdict = 'INSUFFICIENT_DATA'
        reason = '有效算法数 < 3'

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea4_phen2_partial_correlation.json')
    with open(out_json, 'w') as f:
        json.dump({
            'per_algorithm': results,
            'spearman_original': {'rho': rho_spearman, 'p': p_spearman if 'p_spearman' in dir() else None, 'n': len(valid)},
            'partial_correlation': {
                'rho_partial': rho_partial,
                'controlling': 'H_util_total',
                'n': len(valid) if valid else 0,
            },
            'verdict': {'status': verdict, 'reason': reason},
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')
    print(f'\n[VERDICT] {verdict}: {reason}')

    return results, rho_partial, verdict


if __name__ == '__main__':
    main()