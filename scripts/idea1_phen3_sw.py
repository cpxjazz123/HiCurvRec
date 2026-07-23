#!/usr/bin/env python3
"""Idea 1 现象 3: Sliced Wasserstein 交叉验证方向差异

目的:
  - 直接比较 C_1 和 C_2 (单位化后) 与 r_2 (单位化) 的方向分布
  - SW(dir(C_1), dir(r_2)) vs SW(dir(C_2), dir(r_2))
  - 若两者相近 → 没有方向性差异 (进一步坐实尺度假说)

Kill 线:
  - SW(C_1, r_2) 与 SW(C_2, r_2) 差不多 → 没有方向性差异
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import numpy as np
import torch
from scipy.stats import wasserstein_distance

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
DATA = {
    'A_baseline (AQ, K=256^3)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'retrained WF (K=[256,64,16])': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def normalize_per_item(X):
    norms = np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-12)
    return X / norms


def sliced_wasserstein(X, Y, n_projections=64, seed=42):
    """Sliced Wasserstein-1 distance between two sets of unit vectors.
    X: (n, d), Y: (m, d)
    Returns mean over random projections."""
    rng = np.random.default_rng(seed)
    d = X.shape[-1]
    sw_total = 0.0
    for _ in range(n_projections):
        # random direction (uniform on sphere)
        theta = rng.standard_normal(d)
        theta /= np.linalg.norm(theta).clip(1e-12)
        proj_x = X @ theta
        proj_y = Y @ theta
        sw_total += wasserstein_distance(proj_x, proj_y)
    return sw_total / n_projections


def main():
    print('=' * 70)
    print('Idea 1 现象 3 — Sliced Wasserstein 交叉验证')
    print('=' * 70)
    out = {}
    for name, path in DATA.items():
        print(f'\n--- {name} ---')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        codebooks = bundle['codebooks']
        r_lst_np = [r.float().numpy() for r in r_lst]
        cb_np = [c.float().numpy() for c in codebooks]

        # 单位化
        r2_hat = normalize_per_item(r_lst_np[2])
        C0_hat = normalize_per_item(cb_np[0])
        C1_hat = normalize_per_item(cb_np[1])
        C2_hat = normalize_per_item(cb_np[2]) if len(cb_np) > 2 else None

        # 抽样 (SW 在大数据上慢, 用 2000 个 sample)
        n_sample = 2000
        rng = np.random.default_rng(42)
        idx_r = rng.choice(len(r2_hat), size=min(n_sample, len(r2_hat)), replace=False)
        idx_C0 = rng.choice(len(C0_hat), size=min(n_sample, len(C0_hat)), replace=False)
        idx_C1 = rng.choice(len(C1_hat), size=min(n_sample, len(C1_hat)), replace=False)

        r2_s = r2_hat[idx_r]
        C0_s = C0_hat[idx_C0]
        C1_s = C1_hat[idx_C1]

        sw_C0_r2 = sliced_wasserstein(C0_s, r2_s, n_projections=64)
        sw_C1_r2 = sliced_wasserstein(C1_s, r2_s, n_projections=64)
        print(f'  SW(dir(C_1), dir(r̂_2)) = {sw_C0_r2:.6f}')
        print(f'  SW(dir(C_2), dir(r̂_2)) = {sw_C1_r2:.6f}')

        ratio = sw_C0_r2 / (sw_C1_r2 + 1e-12)
        diff = abs(sw_C0_r2 - sw_C1_r2) / ((sw_C0_r2 + sw_C1_r2) / 2 + 1e-12) * 100
        print(f'  比值 (C1/C2) = {ratio:.3f}, 相对差 = {diff:.1f}%')
        if ratio < 1.5:
            kill = '✗ KILL — 方向分布差不多 (没有方向性差异)'
        else:
            kill = '✓ pass — C_1 偏离 r_2 显著大于 C_2'
        print(f'  Kill 判定 (<1.5x): {kill}')

        out[name] = {
            'SW_C0_r2': float(sw_C0_r2),
            'SW_C1_r2': float(sw_C1_r2),
            'ratio': float(ratio),
            'rel_diff_pct': float(diff),
            'kill_verdict': kill,
        }

    out_path = os.path.join(OUT_DIR, 'idea1_phen3_sw.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()