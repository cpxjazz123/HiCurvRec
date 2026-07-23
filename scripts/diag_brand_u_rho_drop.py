#!/usr/bin/env python3
"""Brand U 上的 ρ drop 真实验 — DNC 定义下真正的任务相关方向

目的:
    用 brand (signal-bearing 标签, KNN 2.84× random, 不是 cat_sub 的 0.88× random)
    重新构造 U (top-10 S_B 特征向量), 重算 ρ^{task}(x_c) 和 ρ^{task}(q_1) 对 4+2 算法.

这是 DNC 理论定义下"任务相关"的干净标准 — 在这个 U 上观察到的 ρ drop
才是真正能支撑 "L1 丢失任务信息" 核心 claim 的证据.

输出:
    - ρ^{task}_brand(x_c) 和 ρ^{task}_brand(q_1) per algorithm
    - drop ratio per algorithm
    - 与 cat_sub ρ drop 对比
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import time
import numpy as np
import torch
from collections import Counter

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/brand_u_rho_drop'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
M_TASK = 10
SEED = 42
N_TRIALS = 30

RQIDX = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'idea1_WF':   '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
    'HRQ':        '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/hrq_rqidx.pt',
    'AQ':         '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/aq_rqidx.pt',
}


def build_task_subspace_fast(emb_c_np, labels_str, m=M_TASK, min_count=2):
    """O(K * d² + K³) with rank-(K-1) trick.  min_count filter single-occurrence labels."""
    d = emb_c_np.shape[1]
    cnt = Counter(labels_str)
    valid_labels = sorted([k for k, v in cnt.items() if v >= min_count])
    N = emb_c_np.shape[0]
    mu_G = emb_c_np.mean(axis=0)
    diffs = np.zeros((len(valid_labels), d))
    used_N = 0
    for i, k in enumerate(valid_labels):
        mask = np.array([l == k for l in labels_str])
        n_k = mask.sum()
        if n_k == 0:
            continue
        diffs[i] = np.sqrt(n_k / N) * (emb_c_np[mask].mean(axis=0) - mu_G)
        used_N += n_k
    # Top-m eigenvectors of S_B = diffs^T @ diffs
    S_small = diffs @ diffs.T  # (K, K)
    eigvals, V_small = np.linalg.eigh(S_small)
    idx = np.argsort(-np.abs(eigvals))[:m]
    U = diffs.T @ V_small[:, idx]
    Q, _ = np.linalg.qr(U)
    return Q[:, :m], valid_labels, used_N


def subspace_overlap(U1, U2):
    m = U1.shape[1]
    cos_mat = np.abs(U1.T @ U2)
    return float(cos_mat.max(axis=1).mean())


def rho_task_vec(v, U):
    """ρ^{task}(v) = E[||U^T v||²] / E[||v||²].  v assumed centered per-dim."""
    v_proj = v @ U
    var_task = (v_proj ** 2).sum(axis=1).mean()
    var_total = (v ** 2).sum(axis=1).mean()
    return float(var_task / var_total) if var_total > 0 else 0.0


def main():
    t0 = time.time()
    print('=' * 70)
    print('Brand U 上的 ρ drop 真实验')
    print('=' * 70)
    print(f'  d=2048, m={M_TASK}, N_TRIALS={N_TRIALS}, SEED={SEED}')

    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float().numpy()
    N, d = emb.shape
    print(f'  embedding shape: {emb.shape}')

    with open(META_PATH) as f:
        md = json.load(f)
    labels = {
        'cat_sub': [md[str(i)]['cat_sub'] for i in range(N)],
        'brand':   [md[str(i)]['brand']   for i in range(N)],
    }
    # brand counts
    brand_cnt = Counter(labels['brand'])
    print(f'  brand: {len(brand_cnt)} unique, top: {brand_cnt.most_common(5)}')
    print(f'  brand "Unknown" 占比: {brand_cnt["Unknown"] / N:.3f}')

    emb_centered = emb - emb.mean(axis=0)

    # ===== Step 1: Build U_brand (full data) =====
    print('\n[Step 1] Build U_brand on full data')
    U_brand, valid_labels_brand, used_N = build_task_subspace_fast(emb_centered, labels['brand'], m=M_TASK, min_count=2)
    rho_x_brand = rho_task_vec(emb_centered, U_brand)
    print(f'  U_brand: valid labels={len(valid_labels_brand)}, used_N={used_N}/{N}')
    print(f'  ρ^{{task}}_brand(x_c) = {rho_x_brand:.4f}')

    # ===== Step 2: Build U_cat_sub for comparison =====
    U_cat, valid_labels_cat, _ = build_task_subspace_fast(emb_centered, labels['cat_sub'], m=M_TASK, min_count=2)
    rho_x_cat = rho_task_vec(emb_centered, U_cat)
    print(f'  U_cat_sub: valid labels={len(valid_labels_cat)}')
    print(f'  ρ^{{task}}_cat_sub(x_c) = {rho_x_cat:.4f}')

    # ===== Step 3: Half-split subspace overlap on U_brand =====
    print(f'\n[Step 3] Half-split subspace overlap on U_brand ({N_TRIALS} trials)')
    rng = np.random.default_rng(SEED)
    overlaps_brand = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        perm = rng.permutation(N)
        h1, h2 = perm[:N // 2], perm[N // 2:]
        U1, _, _ = build_task_subspace_fast(emb_centered[h1], [labels['brand'][i] for i in h1], m=M_TASK, min_count=2)
        U2, _, _ = build_task_subspace_fast(emb_centered[h2], [labels['brand'][i] for i in h2], m=M_TASK, min_count=2)
        overlaps_brand[t] = subspace_overlap(U1, U2)
    print(f'  overlap_brand_half_split: {overlaps_brand.mean():.4f} ± {overlaps_brand.std():.4f} '
          f'(z vs random 0.0416 = {(overlaps_brand.mean() - 0.0416) / overlaps_brand.std():.1f})')

    # ===== Step 4: For each algorithm, compute ρ(q_1) on BOTH U_brand and U_cat_sub =====
    print(f'\n[Step 4] ρ^{{task}}_brand(q_1) and ρ^{{task}}_cat_sub(q_1) per algorithm')
    print(f'  {"algorithm":<14} {"ρ_brand(q_1)":<14} {"ρ_brand(x_c)":<14} {"drop_brand":<12} '
          f'{"ρ_cat(q_1)":<12} {"ρ_cat(x_c)":<12} {"drop_cat":<10}')
    rows = []
    for algo, path in RQIDX.items():
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        # Different bundles have different keys; check
        if 'q_lst' in bundle:
            q1 = bundle['q_lst'][0].float().numpy()  # (N, D)
        else:
            raise KeyError(f'{algo} bundle has no q_lst, keys={list(bundle.keys())}')
        # Center q1 per-dim using emb_centered's mu (subtract same global mean)
        q1_centered = q1 - emb.mean(axis=0)
        rho_q1_brand = rho_task_vec(q1_centered, U_brand)
        rho_q1_cat = rho_task_vec(q1_centered, U_cat)
        drop_brand = (rho_x_brand - rho_q1_brand) / rho_x_brand if rho_x_brand > 0 else 0
        drop_cat = (rho_x_cat - rho_q1_cat) / rho_x_cat if rho_x_cat > 0 else 0
        row = {
            'algo': algo,
            'rho_brand_x_c': rho_x_brand,
            'rho_brand_q1': rho_q1_brand,
            'drop_brand': drop_brand,
            'rho_cat_x_c': rho_x_cat,
            'rho_cat_q1': rho_q1_cat,
            'drop_cat': drop_cat,
        }
        rows.append(row)
        print(f'  {algo:<14} {rho_q1_brand:<14.4f} {rho_x_brand:<14.4f} {drop_brand*100:<12.1f}% '
              f'{rho_q1_cat:<12.4f} {rho_x_cat:<12.4f} {drop_cat*100:<10.1f}%')

    # ===== Step 5: Sanity — verify ρ^{task}_brand(x_c) is signal-bearing =====
    print(f'\n[Step 5] Brand subspace sanity — ρ^{{task}}_brand(q_1) should be ≤ ρ^{{task}}_brand(x_c)')
    print(f'  If brand drop > 50% across all algos → "L1 丢 brand 信息" claim has clean evidence')
    print(f'  If brand drop << cat drop → drop is structural (independent of label), not "information loss"')

    # ===== Save =====
    out = {
        'config': {'d': d, 'm': M_TASK, 'N_TRIALS': N_TRIALS, 'SEED': SEED},
        'U_brand_overlap_half_split': {
            'mean': float(overlaps_brand.mean()),
            'std': float(overlaps_brand.std()),
            'median': float(np.median(overlaps_brand)),
        },
        'U_brand_rho_x_c': float(rho_x_brand),
        'U_cat_sub_rho_x_c': float(rho_x_cat),
        'per_algo': rows,
    }
    out_path = os.path.join(OUT_DIR, 'brand_u_rho_drop.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {out_path}')
    print(f'Elapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()