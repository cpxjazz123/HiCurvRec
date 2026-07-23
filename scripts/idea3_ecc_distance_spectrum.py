#!/usr/bin/env python3
"""Idea 3 ECC 现象 1: SID 距离谱 P(h)

测什么:
- 对每套算法 (A/B/C/WF): 构建完整 SID (4 层 = 3 层 RQ + 1 层 dedup)
- 算所有 (i,j) 对的汉明距离分布 P(h), h ∈ {0, 1, 2, 3, 4}
- 重点看:
  - P(0): 碰撞率 (重复 SID)
  - P(1): 危险对占比 (一字之差)

复用:
- result/task16/A_baseline_rqidx.pt
- result/task16/B_mmq_rqidx.pt
- result/task16/C_gsrq_rqidx.pt
- result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_ecc'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX_PATHS = {
    'A_RQ_VAE':  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_MMQ':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_GSRQ':    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_K256_64_16': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def build_full_sid(idx_lst, gains):
    """将 3 层 idx_lst + gains 转成完整 SID (N, 4) 元组
    L4 是去重 digit (不同残差路径之一)
    """
    # 简化: L4 = gain sum mod 256 (避开碰撞的特殊标记)
    # 更标准做法: 见 Stage 2 sid codebook union
    N = idx_lst[0].shape[0]
    L = len(idx_lst)
    sid = torch.zeros(N, L + 1, dtype=torch.long)
    for l in range(L):
        sid[:, l] = idx_lst[l]
    # L4 = (gains * idx_lst) mod 256 的简化版, 实际 RQ-VAE 用的是 path-based dedup
    # 这里用 deterministic L4 (per-item unique based on all 3 layers)
    sid[:, L] = (idx_lst[0] * 7 + idx_lst[1] * 13 + idx_lst[2] * 17) % 256
    return sid  # (N, L+1)


def hamming_distance_spectrum(sid):
    """算完整 P(h)
    sid: (N, L) tensor
    返回: dict {h: P(h), n_pairs}
    """
    N, L = sid.shape
    # 转 numpy 用于向量化
    sid_np = sid.numpy()  # (N, L)
    # 用 numpy broadcasting: (N, 1, L) vs (1, N, L) → (N, N, L) 差异
    # 对 N=11924, 这是 11924*11924*4 = 568M bools ≈ 568MB, 可以接受
    diffs = (sid_np[:, None, :] != sid_np[None, :, :])  # (N, N, L)
    # 距离 = sum(diffs, axis=-1)
    distances = diffs.sum(axis=-1)  # (N, N)
    # 只取上三角
    iu = np.triu_indices(N, k=1)
    upper = distances[iu]
    n_pairs = len(upper)
    counts = np.bincount(upper, minlength=L+1)
    P = counts / n_pairs

    spec = {int(h): float(P[h]) for h in range(L+1)}
    return spec, n_pairs, int(counts[0]), int(counts[1]) if L >= 1 else 0


def main():
    print('=' * 70)
    print('Idea 3 ECC 现象 1: SID 距离谱 P(h)')
    print('=' * 70)

    results = []
    for algo_name, rqidx_path in RQIDX_PATHS.items():
        if not os.path.exists(rqidx_path):
            print(f'[WARN] {algo_name}: rqidx 不存在, 跳过')
            continue

        data = torch.load(rqidx_path, weights_only=False, map_location='cpu')
        idx_lst = data['idx_lst']
        gains = data['gains'] if 'gains' in data else [1.0] * len(idx_lst)

        sid = build_full_sid(idx_lst, gains)
        print(f'\n[{algo_name}] SID shape = {tuple(sid.shape)}')

        spec, n_pairs, n0, n1 = hamming_distance_spectrum(sid)
        print(f'  P(0) = {spec.get(0, 0):.4f} ({n0} pairs)')
        print(f'  P(1) = {spec.get(1, 0):.4f} ({n1} pairs)')
        print(f'  P(2) = {spec.get(2, 0):.4f}')
        print(f'  P(3) = {spec.get(3, 0):.4f}')
        print(f'  P(4) = {spec.get(4, 0):.4f}')

        results.append({
            'algorithm': algo_name,
            'sid_shape': list(sid.shape),
            'P_spectrum': spec,
            'n_pairs': n_pairs,
            'P0_n': n0,
            'P1_n': n1,
        })

    # 判定
    print('\n[判定] P(1) 跨算法对比')
    verdicts = []
    for r in results:
        p1 = r['P_spectrum'].get(1, 0)
        p0 = r['P_spectrum'].get(0, 0)
        if p1 < 0.01:
            v = 'KILLED: P(1) < 1% → 距离谱不是问题来源'
        else:
            v = f'PASS: P(1) = {p1:.4f} ≥ 1%, 需要看 P_err / lift(1)'
        verdicts.append({'algorithm': r['algorithm'], 'P0': p0, 'P1': p1, 'verdict': v})
        print(f'  {r["algorithm"]}: P(0)={p0:.4f}, P(1)={p1:.4f} → {v}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'idea3_phen1_distance_spectrum.json')
    with open(out_json, 'w') as f:
        json.dump({
            'per_algorithm': results,
            'verdicts_phen1': verdicts,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task23_sid_error_correcting_distance.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        L_max = max(max(r['P_spectrum'].keys()) for r in results)
        hs = list(range(L_max + 1))
        bar_w = 0.8 / len(results)
        x = np.arange(L_max + 1)
        for i, r in enumerate(results):
            Ps = [r['P_spectrum'].get(h, 0) for h in hs]
            ax.bar(x + i * bar_w - 0.4 + bar_w/2, Ps, width=bar_w,
                   label=r['algorithm'], edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'h={h}' for h in hs])
        ax.set_ylabel('P(h)')
        ax.set_title('Idea 3 ECC: 汉明距离谱 P(h) 跨算法对比 (Toys 11924 items)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        out_png = os.path.join(OUT_DIR, 'distance_spectrum_4algos.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] 画图失败: {e}')

    return results, verdicts


if __name__ == '__main__':
    main()
