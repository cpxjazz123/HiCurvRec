#!/usr/bin/env python3
"""Regularizer 重启前置探测 — ρ^task brand-LDA 上限估算

目的: 在重启 regularizer 实验之前，估算 brand LDA 子空间方差占比的"理论上限"
      帮助决策重启是否值得。

方法:
  - "完美聚类"基线: 直接用 brand one-hot 编码构造"理想分布"，
    即每个 brand 的均值作为其 class centroid。这种"无量化误差"的分布
    给出的 ρ^task 是 L1 量化器能达到的上限（理论上）。
  - 与 L1 实际达到的 ρ^task = 0.62 比较，给出剩余挖掘空间。

  - 第二个上界: 取 brand centroid + ε噪声（ε → 0 极限），等价于完美聚类
  - 第三个上界: 沿用 L1 量化，但允许"使用 brand centroid 本身"作为 q_1（无误差）

公式:
  ρ^task(v) = ||U^T v||² / ||v||²
  其中 U 是 brand LDA top-10 方向。

理想上限: 把所有商品直接放到对应 brand 的 centroid (mean of items in brand) → q_ideal = centroid_map[brand[i]]
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict

GRID_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID")
OUT_DIR = GRID_ROOT / "result" / "regularizer_upper_bound"
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
M_TASK = 10


def build_brand_subspace(emb_c_np, labels_str, m=10, min_count=2):
    """Fast rank-(K-1) LDA subspace extraction."""
    d = emb_c_np.shape[1]
    cnt = Counter(labels_str)
    valid_labels = sorted([k for k, v in cnt.items() if v >= min_count])
    N = emb_c_np.shape[0]
    mu_G = emb_c_np.mean(axis=0)
    diffs = np.zeros((len(valid_labels), d))
    for i, k in enumerate(valid_labels):
        mask = np.array([l == k for l in labels_str])
        n_k = mask.sum()
        if n_k == 0:
            continue
        diffs[i] = np.sqrt(n_k / N) * (emb_c_np[mask].mean(axis=0) - mu_G)
    S_small = diffs @ diffs.T
    eigvals, V_small = np.linalg.eigh(S_small)
    idx = np.argsort(-np.abs(eigvals))[:m]
    U = diffs.T @ V_small[:, idx]
    Q, _ = np.linalg.qr(U)
    return Q[:, :m]


def rho(v, U):
    """ρ^task(v) = E[||U^T v||²] / E[||v||²]"""
    v_proj = v @ U
    var_task = (v_proj ** 2).sum(axis=1).mean()
    var_total = (v ** 2).sum(axis=1).mean()
    return float(var_task / var_total) if var_total > 0 else 0.0


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=' * 70)
    print('Regularizer 重启前置探测 — ρ^task brand-LDA 上限估算')
    print('=' * 70)

    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    N, d = emb.shape
    print(f'\nEmbedding: {emb.shape}')

    with open(META_PATH) as f:
        md = json.load(f)
    labels_brand = [md[str(i)]['brand'] for i in range(N)]
    print(f'brand labels: {len(labels_brand)}')

    emb_centered = emb - emb.mean(0)

    # Build U_brand
    print('\n[Step 1] Build U_brand on x_centered')
    U_brand = build_brand_subspace(emb_centered, labels_brand, m=M_TASK, min_count=2)
    rho_x = rho(emb_centered, U_brand)
    print(f'  ρ(x_c, U_brand) = {rho_x:.4f}')

    # Method 1: "Perfect clustering" — replace each item with its brand centroid (already centered)
    print('\n[Method 1] Perfect clustering baseline (no quantization error)')
    brand_to_items = defaultdict(list)
    for i, b in enumerate(labels_brand):
        brand_to_items[b].append(i)
    brand_centroids = {}  # brand → (D,) array, mean of items in brand
    for b, items in brand_to_items.items():
        if len(items) >= 2:
            brand_centroids[b] = emb_centered[items].mean(axis=0)
    # Build q_ideal: each item replaced by its brand centroid
    q_ideal = np.zeros_like(emb_centered)
    valid_items = 0
    for i, b in enumerate(labels_brand):
        if b in brand_centroids:
            q_ideal[i] = brand_centroids[b]
            valid_items += 1
    print(f'  valid items (brand count >= 2): {valid_items}/{N}')
    rho_ideal = rho(q_ideal, U_brand)
    print(f'  ρ(q_ideal, U_brand) = {rho_ideal:.4f}  ← THEORETICAL UPPER BOUND')

    # Method 2: brand centroids with shrinkage to global mean
    print('\n[Method 2] Brand centroids with shrinkage to global mean (varying α)')
    rho_shrinkage = []
    for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        q_shrunk = alpha * q_ideal + (1 - alpha) * emb_centered
        rho_s = rho(q_shrunk, U_brand)
        rho_shrinkage.append((alpha, rho_s))
        print(f'  α={alpha:.1f}: ρ(q_shrunk, U_brand) = {rho_s:.4f}')

    # Method 3: A_baseline actual q_1 — for reference
    print('\n[Method 3] A_baseline actual q_1 — for reference')
    a_bundle = torch.load('result/task16/A_baseline_rqidx.pt', map_location='cpu', weights_only=False)
    q1_a = a_bundle['q_lst'][0].float().numpy()
    q1_a_centered = q1_a - q1_a.mean(0)
    rho_a = rho(q1_a_centered, U_brand)
    print(f'  ρ(q_1_A_baseline, U_brand) = {rho_a:.4f}')

    # Method 4: HRQ v2200 actual q_1
    print('\n[Method 4] HRQ_v2200 actual q_1 — for reference')
    hrq_bundle = torch.load('result/diag_joint_cos_f_radial/hrq_rqidx.pt', map_location='cpu', weights_only=False)
    q1_hrq = hrq_bundle['q_lst'][0].float().numpy()
    q1_hrq_centered = q1_hrq - q1_hrq.mean(0)
    rho_hrq = rho(q1_hrq_centered, U_brand)
    print(f'  ρ(q_1_HRQ, U_brand) = {rho_hrq:.4f}')

    # Method 5: AQ actual q_1
    print('\n[Method 5] AQ_additive actual q_1 — for reference')
    aq_bundle = torch.load('result/diag_joint_cos_f_radial/aq_rqidx.pt', map_location='cpu', weights_only=False)
    q1_aq = aq_bundle['q_lst'][0].float().numpy()
    q1_aq_centered = q1_aq - q1_aq.mean(0)
    rho_aq = rho(q1_aq_centered, U_brand)
    print(f'  ρ(q_1_AQ, U_brand) = {rho_aq:.4f}')

    # Save and verdict
    out = {
        'rho_x_c_brand': rho_x,
        'rho_ideal_brand_centroid': rho_ideal,
        'rho_shrinkage': rho_shrinkage,
        'rho_A_baseline': rho_a,
        'rho_HRQ_v2200': rho_hrq,
        'rho_AQ_additive': rho_aq,
    }
    out_path = OUT_DIR / 'upper_bound.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')

    # Decision verdict
    print('\n' + '=' * 70)
    print('DECISION VERDICT')
    print('=' * 70)
    gap_ideal = rho_ideal - rho_aq
    ratio_aq_ideal = rho_aq / rho_ideal if rho_ideal > 0 else 0
    print(f'  ρ(x_c)         = {rho_x:.4f}    (raw embedding brand variance)')
    print(f'  ρ(ideal)       = {rho_ideal:.4f}    (perfect clustering upper bound)')
    print(f'  ρ(A_baseline)  = {rho_a:.4f}    (L1 baseline)')
    print(f'  ρ(HRQ)         = {rho_hrq:.4f}')
    print(f'  ρ(AQ)          = {rho_aq:.4f}    (best L1 currently)')
    print(f'  Gap (ideal - AQ) = {gap_ideal:.4f}')
    print(f'  AQ / ideal     = {ratio_aq_ideal*100:.1f}%')
    if ratio_aq_ideal > 0.90:
        verdict = 'NO_RESTART'
        rationale = 'AQ already achieves >90% of theoretical upper bound — limited room for improvement'
    elif ratio_aq_ideal > 0.75:
        verdict = 'MARGINAL_RESTART'
        rationale = 'AQ achieves 75-90% of upper bound — restart may yield marginal gains, but expensive'
    else:
        verdict = 'RESTART_WORTHWHILE'
        rationale = 'AQ achieves <75% of upper bound — meaningful room to improve, restart recommended (BUT: must use centering-D and brand U, not cat_sub/0.20)'
    print(f'\n  VERDICT: {verdict}')
    print(f'  RATIONALE: {rationale}')
    print('=' * 70)
    print(f'Elapsed: {time.time()-t0:.0f}s')

    # Save verdict
    with open(OUT_DIR / 'verdict.txt', 'w') as f:
        f.write(f'VERDICT: {verdict}\nRATIONALE: {rationale}\n')
        f.write(f'\nρ(x_c)         = {rho_x:.4f}\n')
        f.write(f'ρ(ideal)       = {rho_ideal:.4f}\n')
        f.write(f'ρ(A_baseline)  = {rho_a:.4f}\n')
        f.write(f'ρ(HRQ)         = {rho_hrq:.4f}\n')
        f.write(f'ρ(AQ)          = {rho_aq:.4f}\n')
        f.write(f'Gap (ideal - AQ) = {gap_ideal:.4f}\n')
        f.write(f'AQ / ideal     = {ratio_aq_ideal*100:.1f}%\n')


if __name__ == '__main__':
    main()