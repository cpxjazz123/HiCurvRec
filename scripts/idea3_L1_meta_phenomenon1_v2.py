#!/usr/bin/env python3
"""Idea 3 现象 1 v2: brand-based metadata 关联检验

cat_top 太粗（H=0.0063 bits, 大部分 Unknown）. 改用 Idea B 的 brand 数据
(309 brands ≥5 items, 20 个常用 brand 用于 frequency cutoff).

测:
1. A/B/C 三组 q_l 在 brand label 上的监督信号是否饱和
2. 重点看 L2/L3 的 residual 是否 brand-blind (v1 在 cat_top 上是 blind)
3. 用 brand proxy 取代 cat_top, 提供更高 entropy 的监督目标

判定:
- L2/L3 brand MI 仍 ≈ baseline (0.001) → STACodec 监督信号应放在 L2/L3, 不是 L1
- L2/L3 brand MI 比 baseline 高 → 有 orthogonal 信号可压, Idea 3 改进空间
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import mannwhitneyu
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_L1_aux'
os.makedirs(OUT_DIR, exist_ok=True)

META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
BY_BRAND_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/by_brand.json'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}


def empirical_mi_bits(joint):
    N = joint.sum()
    pxy = joint / N
    px = joint.sum(axis=1) / N
    py = joint.sum(axis=0) / N
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
    return float(mi / np.log(2))


def main():
    print('=' * 70)
    print('Idea 3 现象 1 v2: brand-based metadata MI')
    print('=' * 70)
    metadata = json.load(open(META_PATH))
    by_brand = json.load(open(BY_BRAND_PATH))
    # Build brand label per item
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in
                          range(len(metadata))])
    # Use TOP-N brands, rest = OTHER
    N = len(brand_arr)
    cnt = Counter(brand_arr.tolist())
    # Most frequent brand used as OTHER (NO_BRAND)
    print(f'  total items: {N}')
    print(f'  unique brands (incl. NO_BRAND): {len(cnt)}')
    # Use Top-50 brands, rest = OTHER
    TOP_K = 50
    top_brands = [b for b, _ in cnt.most_common(TOP_K) if b != 'NO_BRAND']
    print(f'  Top-{TOP_K} brands: {len(top_brands)} (excl NO_BRAND)')
    # Cap to top-{TOP_K} + OTHER
    brand_labels = np.array([b if b in top_brands else 'OTHER' for b in brand_arr])
    brand_subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(brand_subs)}
    brand_ids = np.array([brand_to_id[b] for b in brand_labels])

    # H(brand) in bits
    marg = np.bincount(brand_ids)
    p = marg / marg.sum()
    H_brand = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    print(f'  H(brand, {TOP_K}+OTHER) = {H_brand:.4f} bits')

    # === Per-algo ===
    results = {}
    for algo, p in RQIDX_PATHS.items():
        print(f'\n--- {algo} ---')
        bundle = torch.load(p, map_location='cpu', weights_only=False)
        q_lst = bundle['q_lst']
        algo_res = {}
        for l in range(len(q_lst)):
            q_l = q_lst[l].numpy().astype(np.float32)
            km = MiniBatchKMeans(n_clusters=256, random_state=42, n_init=3,
                                 batch_size=1024).fit(q_l)
            c_q = km.labels_.astype(np.int64)
            Kq, Kc = 256, len(brand_subs)
            joint = np.zeros((Kq, Kc), dtype=np.int64)
            np.add.at(joint, (c_q, brand_ids), 1)
            mi_q_brand = empirical_mi_bits(joint)
            nmi = float(normalized_mutual_info_score(c_q, brand_ids))
            rng = np.random.default_rng(42)
            mi_baseline = float(mutual_info_score(c_q, rng.permutation(brand_ids)))
            print(f'  L{l+1}: I(q_l; brand)={mi_q_brand:.4f} bits, NMI={nmi:.4f}, '
                  f'shuf_baseline={mi_baseline:.4f}')
            algo_res[f'l{l+1}'] = {
                'mi_q_brand_bits': mi_q_brand,
                'nmi': nmi,
                'mi_baseline': mi_baseline,
                'frac_H_brand': mi_q_brand / H_brand,
                'signal_ratio': mi_q_brand / (mi_baseline + 1e-12),
            }
        results[algo] = algo_res

    # === Verdict ===
    print('\n' + '=' * 70)
    print('VERDICT — Idea 3 现象 1 v2 (brand)')
    print('=' * 70)
    print(f'  H(brand) = {H_brand:.4f} bits')
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        print(f'\n  {algo}:')
        for lk in ['l1', 'l2', 'l3']:
            v = results[algo][lk]
            print(f'    {lk}: I={v["mi_q_brand_bits"]:.4f} bits '
                  f'({v["frac_H_brand"]:.1%} of H), '
                  f'signal_ratio={v["signal_ratio"]:.2f}x')

    # Check L2/L3 — Are they more blind than L1?
    print('\n  --- Layer ablation: L1 vs L3 brand MI ---')
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        a_l1 = results[algo]['l1']['mi_q_brand_bits']
        a_l3 = results[algo]['l3']['mi_q_brand_bits']
        print(f'    {algo}: L1={a_l1:.4f}, L3={a_l3:.4f}, '
              f'L3/L1 ratio = {(a_l3/(a_l1+1e-12)):.2f}')

    # Kill line: L3 brand MI must exceed baseline+0.005 to count as "有 orthogonal signal"
    a_l3 = results['A_baseline']['l3']
    if a_l3['signal_ratio'] > 5 and a_l3['mi_q_brand_bits'] > 0.05:
        verdict = '✓ L3 有 brand 信号, 监督有意义'
    else:
        verdict = '△ L3 brand 信号弱, 监督可能成本/收益不划算'
    print(f'\n  Idea 3 verdict: {verdict}')

    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    out_path = os.path.join(OUT_DIR, 'idea3_phenomenon1_v2_brand.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'H_brand': H_brand,
            'n_brand_unique': len(brand_subs),
            'top_k': TOP_K,
            'results': results,
            'verdict': verdict,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()