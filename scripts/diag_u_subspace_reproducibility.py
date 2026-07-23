#!/usr/bin/env python3
"""U 子空间半样本复现检验 — 决定性检验 42.5% 是否高维 LDA 过拟合产物

数学优化:
    S_B = Σ_k (n_k/N) (μ_k - μ_G)(μ_k - μ_G)^T
        = M^T M, M_k,: = sqrt(n_k/N) (μ_k - μ_G)        shape (K, 2048)
    top-m eigenvectors of S_B  ⟺  top-m left singular vectors of M
    ⟹ eigh on (K, K) instead of (2048, 2048),  K=24 时几乎瞬时

判别:
    overlap > 0.5 → 真系统性结构, 与判别力弱（KNN 0.88×random）兼容
    overlap 0.15-0.5 → 部分复现, 噪声与信号混合
    overlap < 0.15 → 与随机基线无显著差异, 42.5% 数字崩塌, 原 claim 须撤回
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os
import json
import time
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/u_subspace_reproducibility'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
M_TASK = 10
SEED = 42
N_TRIALS = 30  # 30 trials is plenty for stable mean/std


def build_task_subspace_fast(emb_c_np, labels_str, m=M_TASK):
    """O(K * d² + K³) instead of O(d³)."""
    d = emb_c_np.shape[1]
    label_set = sorted(set(labels_str))
    N = emb_c_np.shape[0]
    mu_G = emb_c_np.mean(axis=0)
    # stack per-class mean diffs
    diffs = np.zeros((len(label_set), d))
    for i, k in enumerate(label_set):
        mask = np.array([l == k for l in labels_str])
        n_k = mask.sum()
        if n_k == 0:
            continue
        # weighted: sqrt(n_k/N) * (mu_k - mu_G)
        diffs[i] = np.sqrt(n_k / N) * (emb_c_np[mask].mean(axis=0) - mu_G)
    # S_B = diffs^T @ diffs (rank ≤ len(label_set)-1)
    # top-m eigenvectors of S_B in span(diffs^T) — project back
    S_small = diffs @ diffs.T  # (K, K)
    eigvals, V_small = np.linalg.eigh(S_small)  # ascending
    # top-m by absolute value
    idx = np.argsort(-np.abs(eigvals))[:m]
    # eigenvectors in R^d = diffs^T @ V_small[:, idx]
    U = diffs.T @ V_small[:, idx]  # (d, m)
    # orthonormalize (numerical safety)
    Q, _ = np.linalg.qr(U)
    return Q[:, :m]


def subspace_overlap(U1, U2):
    """(1/m) Σ_i max_j |<u_i^1, u_j^2>| — principal angle proxy."""
    m = U1.shape[1]
    cos_mat = np.abs(U1.T @ U2)
    return float(cos_mat.max(axis=1).mean())


def main():
    t0 = time.time()
    print('=' * 70)
    print('U 子空间半样本复现检验')
    print('=' * 70)
    print(f'  d=2048, m={M_TASK}, N_TRIALS={N_TRIALS}, SEED={SEED}')

    emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').float().numpy()
    N, d = emb.shape
    print(f'  embedding shape: {emb.shape}, 加载用 {time.time()-t0:.1f}s')

    with open(META_PATH) as f:
        md = json.load(f)
    labels = {
        'cat_sub': [md[str(i)]['cat_sub'] for i in range(N)],
        'cat_top': [md[str(i)]['cat_top'] for i in range(N)],
        'brand':   [md[str(i)]['brand'] for i in range(N)],
    }
    for k, v in labels.items():
        print(f'  {k}: {len(set(v))} unique classes')

    emb_centered = emb - emb.mean(axis=0)
    print(f'  x_centered computed')

    rng = np.random.default_rng(SEED)

    results = {}
    for label_name, label_list in labels.items():
        print(f'\n--- Label: {label_name} ({len(set(label_list))} classes) ---', flush=True)
        overlaps = np.zeros(N_TRIALS)
        # Build U_full once for ρ^{task}(x_c)
        U_full = build_task_subspace_fast(emb_centered, label_list, m=M_TASK)
        rho_full = ((emb_centered @ U_full) ** 2).sum(axis=1).mean() / (emb_centered ** 2).sum(axis=1).mean()
        for t in range(N_TRIALS):
            perm = rng.permutation(N)
            half1, half2 = perm[:N // 2], perm[N // 2:]
            U1 = build_task_subspace_fast(emb_centered[half1], [label_list[i] for i in half1], m=M_TASK)
            U2 = build_task_subspace_fast(emb_centered[half2], [label_list[i] for i in half2], m=M_TASK)
            overlaps[t] = subspace_overlap(U1, U2)
            if (t + 1) % 10 == 0:
                print(f'  trial {t+1}/{N_TRIALS}, overlap_so_far={overlaps[:t+1].mean():.4f}, '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)

        results[label_name] = {
            'n_classes': len(set(label_list)),
            'overlap_mean': float(overlaps.mean()),
            'overlap_std': float(overlaps.std()),
            'overlap_median': float(np.median(overlaps)),
            'overlap_min': float(overlaps.min()),
            'overlap_max': float(overlaps.max()),
            'rho_task_x_centered': float(rho_full),
        }
        print(f'  ✓ {label_name}: overlap={overlaps.mean():.4f} ± {overlaps.std():.4f}, '
              f'ρ^{{task}}(x_c)={rho_full:.4f}')

    # NULL: shuffled cat_sub (real labels, no info)
    print(f'\n--- NULL: cat_sub shuffled (random label, same K) ---', flush=True)
    cat_sub_labels = labels['cat_sub']
    overlaps_null = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        perm = rng.permutation(N)
        half1, half2 = perm[:N // 2], perm[N // 2:]
        shuffled = list(cat_sub_labels)
        rng.shuffle(shuffled)
        U1 = build_task_subspace_fast(emb_centered[half1], [shuffled[i] for i in half1], m=M_TASK)
        U2 = build_task_subspace_fast(emb_centered[half2], [shuffled[i] for i in half2], m=M_TASK)
        overlaps_null[t] = subspace_overlap(U1, U2)
    results['NULL_shuffled_cat_sub'] = {
        'overlap_mean': float(overlaps_null.mean()),
        'overlap_std': float(overlaps_null.std()),
    }
    print(f'  ✓ NULL shuffled: overlap={overlaps_null.mean():.4f} ± {overlaps_null.std():.4f}')

    # NULL: random 10-dim subspaces (metric sanity)
    print(f'\n--- NULL: random 10-dim subspaces (Grassmannian sanity) ---', flush=True)
    overlaps_metric = np.zeros(N_TRIALS)
    for t in range(N_TRIALS):
        A = rng.standard_normal((d, M_TASK))
        Q1, _ = np.linalg.qr(A)
        B = rng.standard_normal((d, M_TASK))
        Q2, _ = np.linalg.qr(B)
        overlaps_metric[t] = subspace_overlap(Q1, Q2)
    results['NULL_random_subspace'] = {
        'overlap_mean': float(overlaps_metric.mean()),
        'overlap_std': float(overlaps_metric.std()),
        'expected_grassmannian_sqrt_md': float(np.sqrt(M_TASK / d)),
    }
    print(f'  ✓ NULL random subspace: overlap={overlaps_metric.mean():.4f} ± {overlaps_metric.std():.4f}  '
          f'(expect ≈ √(m/d) ≈ {np.sqrt(M_TASK/d):.4f})')

    # Save
    out_path = os.path.join(OUT_DIR, 'subspace_reproducibility.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_path}')
    print(f'Total elapsed: {time.time()-t0:.0f}s')

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)
    rand_null = results['NULL_random_subspace']['overlap_mean']
    print(f'  Random subspace null:  {rand_null:.4f} ± {results["NULL_random_subspace"]["overlap_std"]:.4f}')
    print(f'  cat_sub overlap:       {results["cat_sub"]["overlap_mean"]:.4f} ± {results["cat_sub"]["overlap_std"]:.4f}')
    print(f'  cat_top overlap:       {results["cat_top"]["overlap_mean"]:.4f} ± {results["cat_top"]["overlap_std"]:.4f}')
    print(f'  brand overlap:         {results["brand"]["overlap_mean"]:.4f} ± {results["brand"]["overlap_std"]:.4f}')
    print(f'  shuffled cat_sub null: {results["NULL_shuffled_cat_sub"]["overlap_mean"]:.4f}')
    print()

    def judge(name, overlap, std, rand_null):
        z = (overlap - rand_null) / std if std > 0 else 0
        if overlap > 0.5:
            return f'PASS_REPRODUCIBLE ({name} 子空间可复现, z={z:.1f}, 与判别力弱兼容)'
        elif overlap > rand_null + 3 * std:
            return f'PARTIAL ({name} 显著高于随机但远低于 0.5, z={z:.1f}, 信号与噪声混合)'
        else:
            return f'KILL_FOUNDATION ({name} 与随机基线无显著差异, z={z:.1f}, 原 42.5% 数字崩塌)'

    print('  cat_sub:', judge('cat_sub', results['cat_sub']['overlap_mean'],
                            results['cat_sub']['overlap_std'], rand_null))
    print('  cat_top:', judge('cat_top', results['cat_top']['overlap_mean'],
                            results['cat_top']['overlap_std'], rand_null))
    print('  brand:  ', judge('brand',   results['brand']['overlap_mean'],
                            results['brand']['overlap_std'], rand_null))


if __name__ == '__main__':
    main()