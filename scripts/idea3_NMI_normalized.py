#!/usr/bin/env python3
"""Idea 3 补测 1 + 2:
 1) NMI(q_1; brand) 归一化, 对照熵上限判 L1 是否饱和
 2) 训练 raw FLAN-T5 embedding (无 RQ 瓶颈) 线性 probe, 作为无损上限
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.linear_model import LogisticRegression
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_L1_aux'

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}
WF_IDX_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt'


def main():
    print('=' * 70)
    print('Idea 3 补测 1 — NMI(q_l; brand) properly normalized')
    print('Idea 3 补测 2 — raw FLAN-T5 上限 (Train linear probe)')
    print('=' * 70)

    metadata = json.load(open(META_PATH))
    N = len(metadata)

    # Build brand labels (Top-150)
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in range(N)])
    cnt = Counter(brand_arr.tolist())
    TOP_K = 150
    top_brands = [b for b, _ in cnt.most_common(TOP_K) if b != 'NO_BRAND']
    brand_labels = np.array([b if b in top_brands else 'OTHER' for b in brand_arr])
    brand_subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(brand_subs)}
    brand_ids = np.array([brand_to_id[b] for b in brand_labels])
    H_brand = float(-np.sum((p := np.bincount(brand_ids)/N) * np.log2(p + 1e-12)))
    print(f'  N={N}, brand_top_{TOP_K}+OTHER: unique={len(brand_subs)}, H(brand)={H_brand:.4f} bits')

    # === 补测 1: NMI per (algo, layer) ===
    print('\n--- NMI(q_l; brand) 归一化 (KMeans cluster assignment 协议) ---')
    RQIDX_PATHS_FULL = {**RQIDX_PATHS, 'idea1_WF': WF_IDX_PATH}
    nmi_results = {}
    for algo, path in RQIDX_PATHS_FULL.items():
        if not os.path.exists(path):
            continue
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        idx_lst = bundle['idx_lst']
        q_lst = bundle['q_lst']
        algo_res = {}
        for l in range(len(idx_lst)):
            # Use the actual code indices (cluster ids) as the discrete label
            c_q = idx_lst[l].numpy()
            # H(q_l) bits
            marg = np.bincount(c_q)
            pq = marg / marg.sum()
            H_q = float(-np.sum(pq[pq>0] * np.log2(pq[pq>0])))
            # MI(q_l; brand)
            mi_raw = float(mutual_info_score(c_q, brand_ids))
            mi_bits = mi_raw / np.log(2)
            # Normalized MI = MI / mean(H(q), H(brand))  (symmetric normalization)
            denom = (H_q + H_brand) / 2 + 1e-12
            nmi = mi_bits / denom
            print(f'  {algo} L{l+1}: H(q_l)={H_q:.4f}, H(brand)={H_brand:.4f}, '
                  f'I(q_l; brand)={mi_bits:.4f} bits, NMI={nmi:.4f}')
            algo_res[f'l{l+1}'] = {
                'H_q': H_q,
                'H_brand': H_brand,
                'MI_bits': mi_bits,
                'NMI': nmi,
            }
        nmi_results[algo] = algo_res

    # === 补测 2: raw FLAN-T5 无损上限 ===
    print('\n--- raw FLAN-T5 上限 — Linear probe on x (无 RQ 瓶颈) ---')
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    # Use 80/20 split
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    train_idx = perm[:int(N*0.8)]
    test_idx = perm[int(N*0.8):]
    # Logistic regression (linear) — could use larger but linear is the "upper bound" for a single layer
    lr = LogisticRegression(
        C=1.0, max_iter=200, multi_class='multinomial', solver='lbfgs',
        n_jobs=4, random_state=42,
    )
    # Subsample for speed (full 11924×2048 may take minutes)
    SUB = 8000
    sub_idx = rng.choice(N, size=SUB, replace=False)
    lr.fit(x[sub_idx], brand_ids[sub_idx])
    pred_test = lr.predict(x[test_idx])
    acc = float((pred_test == brand_ids[test_idx]).mean())
    # Also calibration: train acc
    pred_train = lr.predict(x[train_idx])
    acc_train = float((pred_train == brand_ids[train_idx]).mean())
    # Majority class baseline
    top_class = float(np.bincount(brand_ids).max() / N)
    print(f'  Linear probe on x (raw FLAN-T5 embedding):')
    print(f'    train_acc={acc_train:.4f}, test_acc={acc:.4f}, '
          f'top_class_baseline={top_class:.4f}, ratio={acc/top_class:.2f}x')
    upper_bound = {
        'test_acc_linear_probe': acc,
        'train_acc_linear_probe': acc_train,
        'top_class_baseline_acc': top_class,
        'ratio_to_top_class': acc/top_class,
    }

    # Also: K-Means cluster on raw x with K_use to make NMI
    K_use = 256
    km = MiniBatchKMeans(n_clusters=K_use, random_state=42, n_init=3, batch_size=1024).fit(x)
    c_x = km.labels_.astype(np.int64)
    nmi_x_brand = float(normalized_mutual_info_score(c_x, brand_ids))
    print(f'  K-Means on raw x (K={K_use}): NMI(x_cluster; brand)={nmi_x_brand:.4f}')

    # === 判定 ===
    print('\n' + '=' * 70)
    print('VERDICT — Idea 3 补测 1+2')
    print('=' * 70)
    # 阈值
    print('  NMI threshold per user: >0.8 → 接近上限 → KILL')
    print('                         <0.5 → 还有空间 → 不能直接判死')
    for algo in nmi_results:
        a = nmi_results[algo]
        for lk in ['l1', 'l2', 'l3']:
            if lk in a:
                nmi = a[lk]['NMI']
                print(f'    {algo} {lk}: NMI={nmi:.4f}')

    # Compare L1 NMI to upper bound
    if 'A_baseline' in nmi_results:
        a_l1 = nmi_results['A_baseline']['l1']['NMI']
        a_l3 = nmi_results['A_baseline']['l3']['NMI']
        print(f'\n  A_baseline L1 NMI = {a_l1:.4f}')
        print(f'  A_baseline L3 NMI = {a_l3:.4f}')
        print(f'  raw K-Means upper NMI = {nmi_x_brand:.4f}')
        # Compute L1 vs upper
        gap_l1 = nmi_x_brand - a_l1
        gap_l3 = nmi_x_brand - a_l3
        print(f'  Gap L1 → upper: {gap_l1:+.4f}')
        print(f'  Gap L3 → upper: {gap_l3:+.4f}')
        if gap_l1 < 0.05:
            verdict_l1 = '✓ L1 NMI 已接近上限 (KILL — L1 饱和)'
        elif gap_l1 > 0.3:
            verdict_l1 = '✗ L1 仍有大幅空间, 不能判死 (但 L3 如何是另一个故事)'
        else:
            verdict_l1 = '△ L1 中等 (case-by-case)'
        print(f'  {verdict_l1}')

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

    out_path = os.path.join(OUT_DIR, 'idea3_NMI_normalized.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'H_brand': H_brand,
            'K_use_KMeans': K_use,
            'nmi_per_algo_layer': nmi_results,
            'upper_bound_linear_probe': upper_bound,
            'upper_bound_KMeans_NMI': nmi_x_brand,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()