#!/usr/bin/env python3
"""Idea 3 现象 1: q_1 与 metadata 关联检验

对 A_baseline:
1. q_1 是 L1 codebook 输出, shape (N, D=2048)
2. 测 q_1 与 cat_top 的互信息 (cat_top 太粗, 只有 5 个非 Unknown 类)
3. 测 q_1 与 PPMI 协同 embedding 的相关性 (高/低协同对在 q_1 空间距离差异)
4. 比较 q_1 与 q_2 / q_3 (看 L1 是不是真的"抓到"了粗粒度语义)

判定:
- q_1 与 metadata 关联弱 → L1 有改进空间, 监督有意义
- q_1 与 metadata 关联已经很强 → L1 已抓够, 监督是 trivial
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# Pre-import to break circular import
import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import mannwhitneyu

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_L1_aux'
os.makedirs(OUT_DIR, exist_ok=True)

CF_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/cf_ppmi_svd256.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
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
    print('Loading metadata + CF + r_lst')
    print('=' * 70)
    metadata = json.load(open(META_PATH))
    cf = torch.load(CF_PATH, map_location='cpu', weights_only=False)
    f_norm = cf['f_normalized'].numpy().astype(np.float32)
    N = f_norm.shape[0]
    print(f'  f shape: {f_norm.shape}')

    # Cat_top labels per item
    cat_top_arr = np.array([metadata.get(str(i), {}).get('cat_top', 'Unknown') for i in range(N)])
    cat_subs = sorted(set(cat_top_arr))
    cat_to_id = {c: i for i, c in enumerate(cat_subs)}
    cat_ids = np.array([cat_to_id[c] for c in cat_top_arr])
    print(f'  unique cat_top: {len(cat_subs)}: {cat_subs[:6]}...')

    # H(cat) in bits
    cat_marg = np.bincount(cat_ids)
    cat_p = cat_marg / cat_marg.sum()
    H_cat = -float(np.sum(cat_p[cat_p > 0] * np.log2(cat_p[cat_p > 0])))
    print(f'  H(cat_top) = {H_cat:.4f} bits')

    # === Per-algo: q_l vs metadata ===
    results = {}
    for algo, p in RQIDX_PATHS.items():
        print(f'\n--- {algo} ---')
        bundle = torch.load(p, map_location='cpu', weights_only=False)
        q_lst = bundle['q_lst']
        r_lst = bundle['r_lst']
        algo_res = {}
        for l in range(len(q_lst)):
            q_l = q_lst[l].numpy().astype(np.float32)  # (N, 2048)
            # K-Means cluster q_l into K=256 codes (mimicking SID codebook)
            km = MiniBatchKMeans(n_clusters=256, random_state=42, n_init=3,
                                 batch_size=1024).fit(q_l)
            c_q = km.labels_.astype(np.int64)
            # Joint (c_q, cat) for empirical MI
            Kq, Kc = 256, len(cat_subs)
            joint = np.zeros((Kq, Kc), dtype=np.int64)
            np.add.at(joint, (c_q, cat_ids), 1)
            mi_q_cat = empirical_mi_bits(joint)
            # Normalized MI
            mi_nats = float(mutual_info_score(c_q, cat_ids))
            nmi = float(normalized_mutual_info_score(c_q, cat_ids))
            # Random baseline
            rng = np.random.default_rng(42)
            mi_baseline = float(mutual_info_score(c_q, rng.permutation(cat_ids)))
            print(f'  L{l+1} q: I(q_l; cat)={mi_q_cat:.4f} bits, NMI={nmi:.4f}, '
                  f'shuf_baseline={mi_baseline:.4f}')

            # CF pair distance (high vs low sim_cf)
            sim_cf_diag = f_norm @ f_norm.T  # (N, N) — too large to compute fully; subsample
            rng = np.random.default_rng(42)
            n_pairs = 30000
            i_idx = rng.integers(0, N, size=n_pairs)
            j_idx = rng.integers(0, N, size=n_pairs)
            valid = i_idx != j_idx
            i_idx, j_idx = i_idx[valid], j_idx[valid]
            sim_pairs = (f_norm[i_idx] * f_norm[j_idx]).sum(axis=1)
            tau_high = float(np.quantile(sim_pairs, 0.90))
            tau_low = float(np.quantile(sim_pairs, 0.10))
            high_mask = sim_pairs > tau_high
            low_mask = sim_pairs < tau_low
            # In q_l space: ‖q_l[i] - q_l[j]‖
            dist_high = np.linalg.norm(q_l[i_idx[high_mask]] - q_l[j_idx[high_mask]], axis=1)
            dist_low  = np.linalg.norm(q_l[i_idx[low_mask]]  - q_l[j_idx[low_mask]],  axis=1)
            delta_q = float(dist_low.mean() - dist_high.mean())
            pooled = np.sqrt((dist_high.var() + dist_low.var()) / 2 + 1e-12)
            cohens_d = delta_q / pooled
            mw_u, mw_p = mannwhitneyu(dist_low, dist_high, alternative='greater')
            print(f'    CF pair distance Δ={delta_q:+.4f}, d={cohens_d:.3f}, MW-p={mw_p:.2e}')

            algo_res[f'l{l+1}'] = {
                'mi_q_cat_bits': mi_q_cat,
                'nmi': nmi,
                'mi_baseline': mi_baseline,
                'delta_cf_pair_dist': delta_q,
                'cohens_d': cohens_d,
                'mw_p': float(mw_p),
                'sig': bool(mw_p < 0.01 and cohens_d > 0.1),
            }
        results[algo] = algo_res

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea 3 现象 1')
    print('=' * 70)
    # Compare A_baseline L1
    a_l1 = results['A_baseline']['l1']
    frac_H_cat = a_l1['mi_q_cat_bits'] / H_cat
    print(f'  A_baseline L1: I(q_1; cat_top) = {a_l1["mi_q_cat_bits"]:.4f} bits '
          f'({frac_H_cat:.1%} of H(cat)={H_cat:.4f})')
    print(f'  A_baseline L1: CF pair Δdist = {a_l1["delta_cf_pair_dist"]:+.4f}, d = {a_l1["cohens_d"]:.3f}')
    # Kill line: I(q_1; cat) ≈ H(cat) (L1 已经抓到上限)
    if frac_H_cat > 0.85:
        verdict_l1 = '✗ L1 已近 cat 上限 (kill)'
    elif a_l1['mi_q_cat_bits'] < 0.5 * H_cat:
        verdict_l1 = '✓ L1 远未饱和 (有改进空间)'
    else:
        verdict_l1 = '△ L1 中等饱和'
    print(f'  L1 verdict: {verdict_l1}')

    # Also check cross-layer progression
    print('\n  Cross-layer q_1 → q_2 → q_3 I(q; cat):')
    for lk in ['l1', 'l2', 'l3']:
        v = results['A_baseline'][lk]['mi_q_cat_bits']
        print(f'    {lk}: {v:.4f}')

    out_path = os.path.join(OUT_DIR, 'idea3_phenomenon1.json')

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

    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'H_cat': H_cat,
            'n_cat_unique': len(cat_subs),
            'results': results,
            'L1_verdict': verdict_l1,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()