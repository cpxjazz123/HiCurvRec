#!/usr/bin/env python3
"""Idea 1 现象 4: 范数衰减 vs V-info 衰减相关性

目的:
  - 测逐层 E[‖r_l‖] 和 erank_l
  - 与 task17_v_info 的 V-info(l) 做相关性分析
  - 看"第一层 V-info 高"是否本质上 = "第一层残差范数大, 承载大部分总方差"

若两者高度相关 → 把"结构性断裂"降级为"残差方差主要集中在浅层"
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import numpy as np
import torch
from scipy.stats import spearmanr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
DATA = {
    'A_baseline (AQ, K=256^3)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'retrained WF (K=[256,64,16])': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def effective_rank(X):
    """erank = exp(H(p)), p_i = σ_i / sum σ (归一化奇异值谱)."""
    s = np.linalg.svd(X, compute_uv=False)
    p = s / s.sum().clip(1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))


def per_layer_norm_stats(r_lst_np, cb_np):
    """测每层:
      E[‖r_l‖] (l=0,1,2,...)  input residual norm
      E[‖c^(l)‖]              codeword norm
      erank(r_l)              input effective rank
      erank(c^(l))            codeword effective rank
    """
    out = {'r': [], 'c': [], 'erank_r': [], 'erank_c': []}
    for l in range(len(r_lst_np)):
        r = r_lst_np[l]
        out['r'].append(float(np.linalg.norm(r, axis=-1).mean()))
        out['erank_r'].append(effective_rank(r))
    for l in range(len(cb_np)):
        c = cb_np[l]
        out['c'].append(float(np.linalg.norm(c, axis=-1).mean()))
        out['erank_c'].append(effective_rank(c))
    return out


def main():
    print('=' * 70)
    print('Idea 1 现象 4 — 范数衰减 vs V-info 衰减')
    print('=' * 70)
    out = {}
    for name, path in DATA.items():
        print(f'\n--- {name} ---')
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        codebooks = bundle['codebooks']
        r_lst_np = [r.float().numpy() for r in r_lst]
        cb_np = [c.float().numpy() for c in codebooks]
        stats = per_layer_norm_stats(r_lst_np, cb_np)

        print(f'  Per-layer norms:')
        for l in range(len(stats['r'])):
            print(f'    l={l}: E[‖r_l‖]={stats["r"][l]:.4f}, erank(r_l)={stats["erank_r"][l]:.2f}')
        for l in range(len(stats['c'])):
            print(f'    l={l}: E[‖c^(l)‖]={stats["c"][l]:.4f}, erank(c^(l))={stats["erank_c"][l]:.2f}')

        # 从 task19 input_entropy 读 V-info proxy (就是 input to layer l 的 cluster entropy)
        # 这里重新算, 因为 input_entropy_per_layer 来自 idea1_delta_L_diagnostic
        from idea1_delta_L_diagnostic import compute_deltas
        # 也用 cluster entropy 作为 V-info proxy
        from sklearn.cluster import MiniBatchKMeans
        from collections import Counter
        v_info = []
        for l in range(1, len(r_lst_np)):
            r = r_lst_np[l]
            km = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3, batch_size=1024).fit(r)
            c = km.labels_.astype(np.int64)
            cnt = Counter(c.tolist())
            p = np.array([v for v in cnt.values()]) / len(c)
            H = float(-(p * np.log2(p + 1e-12)).sum())
            v_info.append(H)

        print(f'  V-info proxy (cluster H per layer input):')
        for l, h in enumerate(v_info):
            print(f'    input_to_layer_{l+1}: H={h:.4f} bits')

        # Spearman corr: E[‖r_l‖] vs V-info(l)
        if len(v_info) >= 3:
            r_norms_l1_3 = stats['r'][1:1+len(v_info)]  # r_1, r_2, r_3 对应 V-info(1), V-info(2), V-info(3)
            sp_r, p_r = spearmanr(r_norms_l1_3, v_info)
            print(f'  Spearman corr(E[‖r_l‖], V-info(l)) = {sp_r:.4f} (p={p_r:.4e})')

            # E[‖c^(l)‖] vs V-info (用 l=0,1,2)
            c_norms = stats['c'][:len(v_info)]
            sp_c, p_c = spearmanr(c_norms, v_info)
            print(f'  Spearman corr(E[‖c^(l)‖], V-info(l)) = {sp_c:.4f} (p={p_c:.4e})')

            # erank(r_l) vs V-info
            erank_r = stats['erank_r'][1:1+len(v_info)]
            sp_e, p_e = spearmanr(erank_r, v_info)
            print(f'  Spearman corr(erank(r_l), V-info(l)) = {sp_e:.4f} (p={p_e:.4e})')

            out[name] = {
                'r_norms': stats['r'],
                'c_norms': stats['c'],
                'erank_r': stats['erank_r'],
                'erank_c': stats['erank_c'],
                'v_info_proxy': v_info,
                'spearman_r_norm_vs_vinfo': float(sp_r),
                'spearman_c_norm_vs_vinfo': float(sp_c),
                'spearman_erank_vs_vinfo': float(sp_e),
                'p_values': {'r': float(p_r), 'c': float(p_c), 'erank': float(p_e)},
            }
        else:
            out[name] = {'r_norms': stats['r'], 'c_norms': stats['c'],
                        'v_info_proxy': v_info}

    out_path = os.path.join(OUT_DIR, 'idea1_phen4_norm_vinfo.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()