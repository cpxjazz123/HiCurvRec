#!/usr/bin/env python3
"""Idea 3 补测 3 — Post-hoc brand 残差注入

测深层监督的理论上限:
- 对 layer l 的残差 r_l, 注入 brand-centroid mean 残差
- 测 I(injected_r_l; brand) vs I(original_r_l; brand) → ΔI_V^{aux-deep}
- 大 → 残差空间有 brand 信号可注入 → L2/L3 监督理论上能学 brand
- 小 → 残差空间已被 L1 抽干 → L2/L3 监督 ROI 低

用户原 IDEA: 让 L2/L3 监督 brand 信号, 但 L1 已经学到 brand = saturation

判定:
- ΔI_V^{aux-deep} 大 + L1 仍饱和 → KILL (L1 已最优, L2/L3 重复劳动)
- ΔI_V^{aux-deep} 小 → KILL (残差空间没 brand 信号, 监督无效)
- 任何情况 → Idea 3 KILL
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from sklearn.decomposition import PCA

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea3_L1_aux'

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'

IDX_FILES = {
    'A_baseline (AQ K=256^3)':   '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq (cosine K=256^3)':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq (gain-shape K=256^3)':'/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF (K=[256,64,16])':         '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}


def fast_mi_bits(x, y, n_sample=2000, seed=42, n_neighbors=5):
    """KSG via sklearn + PCA-16 reduction."""
    if x.ndim == 2 and x.shape[1] > 32:
        x = PCA(n_components=16, random_state=seed).fit_transform(x)
    N = x.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    mi_nats = mutual_info_classif(x_s, y_s, n_neighbors=n_neighbors,
                                   random_state=seed, discrete_features=False, n_jobs=4)
    return float(mi_nats.mean() / np.log(2))


def build_brand_centroids(r_l, brand_ids):
    """Per-brand mean residual vector."""
    unique_brands = np.unique(brand_ids)
    brand_to_idx = {b: i for i, b in enumerate(unique_brands)}
    centroids = np.zeros((len(unique_brands), r_l.shape[1]), dtype=np.float32)
    for b in unique_brands:
        mask = brand_ids == b
        centroids[brand_to_idx[b]] = r_l[mask].mean(axis=0)
    return centroids, brand_to_idx


def inject_brand_residual(r_l, brand_ids, alpha):
    """Post-hoc injection: r_l' = r_l + alpha * (centroid_brand - centroid_overall)."""
    centroids, b2i = build_brand_centroids(r_l, brand_ids)
    # Centroid of overall
    overall = r_l.mean(axis=0, keepdims=True)
    # Per-sample brand-deviation from overall
    deviation = centroids[ np.array([b2i[b] for b in brand_ids]) ] - overall  # (N, D)
    return r_l + alpha * deviation


def main():
    print('=' * 70)
    print('Idea 3 补测 3 — Post-hoc brand 残差注入 → ΔI_V^{aux-deep}')
    print('=' * 70)

    metadata = json.load(open(META_PATH))
    N = len(metadata)
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in range(N)])
    cnt = Counter(brand_arr.tolist())
    TOP_K = 150
    top_brands = [b for b, _ in cnt.most_common(TOP_K) if b != 'NO_BRAND']
    brand_labels = np.array([b if b in top_brands else 'OTHER' for b in brand_arr])
    brand_subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(brand_subs)}
    brand_ids = np.array([brand_to_id[b] for b in brand_labels])
    print(f'  N={N}, unique brand labels={len(brand_subs)}')

    # Sweep alpha
    alphas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    # Per-algo, per-layer ΔI_V
    all_results = {}
    for algo, path in IDX_FILES.items():
        print(f'\n--- {algo} ---')
        if not os.path.exists(path):
            print(f'  FILE NOT FOUND: {path}, skip')
            continue
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        r_lst = bundle['r_lst']
        algo_res = {}
        # r_lst[0] = x, r_lst[1..L] = input to layer l
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            mi_orig = fast_mi_bits(r_l, brand_ids)
            print(f'  Layer {l} input (residual): I(r_l; brand) original = {mi_orig:.4f} bits')
            mi_per_alpha = {}
            for a in alphas:
                r_l_inj = inject_brand_residual(r_l, brand_ids, alpha=a)
                mi_inj = fast_mi_bits(r_l_inj, brand_ids)
                delta = mi_inj - mi_orig
                mi_per_alpha[float(a)] = {
                    'mi_injected': mi_inj,
                    'delta': delta,
                }
                print(f'    alpha={a:.1f}: I(injected; brand)={mi_inj:.4f}, '
                      f'ΔI_V^{{aux-deep}}={delta:+.4f}')
            algo_res[f'l{l}_input'] = {
                'mi_original': mi_orig,
                'mi_per_alpha': mi_per_alpha,
            }
        all_results[algo] = algo_res

    # === 判定 ===
    print('\n' + '=' * 70)
    print('VERDICT — Idea 3 补测 3 (深层 brand 监督理论上限)')
    print('=' * 70)
    for algo, res in all_results.items():
        for lk, data in res.items():
            mi_orig = data['mi_original']
            best_alpha = max(data['mi_per_alpha'].keys(),
                             key=lambda a: data['mi_per_alpha'][a]['delta'])
            best_delta = data['mi_per_alpha'][best_alpha]['delta']
            print(f'  {algo} {lk}: I_orig={mi_orig:.4f}, '
                  f'best ΔI_V (alpha={best_alpha})={best_delta:+.4f}')

    # Key question: 即使注入到 L2/L3 的 brand 信号能提多少?
    # ΔI_V^{aux-deep} = I(injected; brand) - I(orig; brand)
    # 这是 post-hoc injection 上限 — 即"如果 L2/L3 学到完美的 brand-centroid 监督,
    # 在残差空间能拿到多少 brand 信息"
    # 与 L1 已经学到的信息对比 → 如果 L2/L3 ΔI_V 比 L1 NMI 还小, 监督没必要

    print('\n  Insight: ΔI_V^{aux-deep} 是深层 brand 监督的 post-hoc 上限。')
    print('  - 如果 ΔI_V^{aux-deep} 在所有层都小, 说明残差空间没有 brand 信号')
    print('    → L2/L3 brand 监督必然无效 (没有可学的信号)')
    print('  - 如果 ΔI_V^{aux-deep} 大, 但 L1 已经 NMI=0.5156 接近 raw 上限 0.5351')
    print('    → 把 brand 信号从 L1 移到 L2/L3 是"重复劳动" + 损失其他信号')

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

    out_path = os.path.join(OUT_DIR, 'idea3_posthoc_brand_injection.json')
    with open(out_path, 'w') as f:
        json.dump(json_safe({
            'alphas': alphas,
            'n_unique_brand_labels': len(brand_subs),
            'per_algo_layer': all_results,
        }), f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()