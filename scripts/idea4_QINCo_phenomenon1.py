#!/usr/bin/env python3
"""Idea 4 现象 1: QINCo dynamic codebook 价值检验

QINCo 核心思想: 用 input-conditioned dynamic codebook 替代 static centroid.
如果 cluster 内 residual 方差很大 (同类样本仍分散), 静态 centroid 必然损失,
dynamic 应该更优.

测:
- 对每个 L1 cluster, 算 cluster 内 residual 方差 trace(Σ_cluster)
- 比 cluster 间 residual 方差: ratio = within / between
- 低 ratio (within << between) → 静态代码本够用, QINCo 收益有限
- 高 ratio (within ≈ between) → 静态代码本丢信息, QINCo 应该有效

判定:
- ratio < 0.3 → ✗ QINCo 收益低 (kill)
- ratio > 0.7 → ✓ QINCo 收益高 (值得做 Idea 4 现象 2)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea4_QINCo_aux'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}


def main():
    print('=' * 70)
    print('Idea 4 现象 1 — Within/Between cluster variance ratio')
    print('=' * 70)

    for algo, p in RQIDX_PATHS.items():
        print(f'\n--- {algo} ---')
        bundle = torch.load(p, map_location='cpu', weights_only=False)
        idx_lst = bundle['idx_lst']
        r_lst = bundle['r_lst'][1:]  # 跳过 r[0]=x, 用 r[1..3] 是 layer input 的 residual
        q_lst = bundle['q_lst']

        for l in range(3):
            idx = idx_lst[l].numpy()  # (N,) cluster ids
            r = r_lst[l].numpy().astype(np.float32)  # (N, 2048) layer input
            K = idx.max() + 1
            # Within-cluster variance: for each cluster c, var_{i in c}(r_i)
            # Between-cluster variance: var_c(mean_c)
            r_mean_total = r.mean(axis=0)  # (D,)
            ss_total = ((r - r_mean_total) ** 2).sum()  # N * trace(Σ_total)
            ss_within = 0.0
            ss_between = 0.0
            cluster_sizes = []
            for c in range(K):
                mask = idx == c
                n_c = mask.sum()
                if n_c < 2:
                    continue
                cluster_sizes.append(int(n_c))
                r_c = r[mask]
                r_c_mean = r_c.mean(axis=0)
                ss_within += ((r_c - r_c_mean) ** 2).sum()
                ss_between += n_c * ((r_c_mean - r_mean_total) ** 2).sum()
            total = ss_within + ss_between
            ratio = float(ss_within / (ss_within + ss_between + 1e-12))
            # Effective rank of between-cluster mean matrix (skip)
            cluster_n = np.array(cluster_sizes)
            print(f'  L{l+1}: K={K} (active {len(cluster_sizes)}), '
                  f'sizes min={cluster_n.min()}, max={cluster_n.max()}, '
                  f'mean={cluster_n.mean():.1f}')
            print(f'    ratio within/total = {ratio:.4f}')
            print(f'    within SS = {ss_within:.2e}, between SS = {ss_between:.2e}')

            if l == 0:
                # Save per-layer metric into OUT
                pass

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)
    print('  0 < ratio < 1: lower → 静态代码本够用')
    print('  参考: 完全随机分配 cluster 期望 ratio ≈ (1 - 1/K) ≈ 0.996')
    print('         完美聚类 ratio → 0.0')
    print()
    print('  阈值经验:')
    print('    ratio < 0.3 → 强聚类, static OK (QINCo 收益小)')
    print('    ratio 0.3-0.6 → 中等, QINCo 可能有效')
    print('    ratio > 0.7 → 弱聚类, QINCo 应该有效')


if __name__ == '__main__':
    main()