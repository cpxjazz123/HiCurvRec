#!/usr/bin/env python3
"""Idea A 现象 1: FastICA 后独立性检验 (最便宜的总闸)

对比:
  - 原始 embedding x ∈ R^{N × 2048}
  - FastICA s = W_ICA · (x - mean)  (200 维, sklearn FastICA 限制)

测两两成分间 MI (用 sklearn KSG-based mutual_info_classif, PCA 16-dim 加速)
对比 kill 线: ICA 后 MI ≈ 原始 MI (说明 flan-t5 已经接近独立轴)
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json, time
import numpy as np
import torch
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import mutual_info_score

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaA_ica_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 4000      # ICA 拟合子样本 (FastICA 对 N 很敏感)
N_PAIRS_PER_PAIR = 2000 # 用于测两两 MI 的对数 (避免 N=11924 太慢)
PCA_MI_DIM = 16         # 用于测 MI 时的 PCA 降维
N_ICA_COMP = 200        # FastICA 成分数 (200 维就够, 2048 全跑太慢)


def fast_mi_bits_pair(x, y, n_bins=20, seed=42):
    """MI between two 1D arrays via rank-bin + sklearn.mutual_info_score.

    Quantile-based binning into n_bins discrete labels, then empirical MI in nats/bits.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    N = len(x)
    if N > 2000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=2000, replace=False)
        x, y = x[idx], y[idx]
    # Quantile bin
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins - 1))[1:-1])
    yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins - 1))[1:-1])
    mi_nats = mutual_info_score(xb, yb)
    return float(mi_nats / np.log(2))


def kurtosis_batch(X):
    """kurtosis (excess) per column."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    var = (Xc ** 2).mean(axis=0)
    std = np.sqrt(var) + 1e-12
    kurt = ((Xc / std) ** 4).mean(axis=0) - 3.0
    return kurt


def main():
    print('=' * 70)
    print('Loading Stage 1 embedding x ∈ R^{11924 × 2048}')
    print('=' * 70)
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    print(f'  x shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}')

    # Subsample for ICA fit
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(x.shape[0], size=N_SUBSAMPLE, replace=False)
    x_sub = x[idx_sub]

    # PCA prewhiten for FastICA stability
    print(f'\nPre-whitening with PCA(n_components={N_ICA_COMP})...')
    t0 = time.time()
    pca_pre = PCA(n_components=N_ICA_COMP, random_state=42, whiten=True)
    x_sub_white = pca_pre.fit_transform(x_sub)
    print(f'  shape={x_sub_white.shape}, time={time.time()-t0:.1f}s')

    print(f'\nFit FastICA on {N_SUBSAMPLE} subsample (n_components={N_ICA_COMP})...')
    t0 = time.time()
    ica = FastICA(n_components=N_ICA_COMP, algorithm='parallel',
                  whiten='unit-variance', max_iter=500, tol=1e-3,
                  random_state=42)
    s_sub = ica.fit_transform(x_sub_white)
    print(f'  s shape={s_sub.shape}, time={time.time()-t0:.1f}s')

    # Save ICA model + transform full set
    print('\nTransform full x via PCA → FastICA pipeline...')
    t0 = time.time()
    x_white_full = pca_pre.transform(x)
    s_full = ica.transform(x_white_full)
    print(f'  s_full shape={s_full.shape}, time={time.time()-t0:.1f}s')

    # Save for downstream (Idea A 现象 2)
    np.savez(os.path.join(OUT_DIR, 'ica_model.npz'),
             pca_components=pca_pre.components_,
             pca_mean=pca_pre.mean_,
             ica_components=ica.components_,
             ica_mean=ica.mean_,
             ica_whiten=ica.whitening_,
             ica_mixing=ica.mixing_)
    np.save(os.path.join(OUT_DIR, 's_full_ica200.npy'), s_full.astype(np.float32))
    print(f'  Saved s_full_ica200.npy (shape={s_full.shape})')

    # Kurtosis comparison
    print('\n' + '=' * 70)
    print('Kurtosis: 原始 x (PCA 前 200 维) vs ICA s')
    print('=' * 70)
    x_pca200 = pca_pre.transform(x)         # (N, 200) whitened PCA of x
    kurt_x = kurtosis_batch(x_pca200)
    kurt_s = kurtosis_batch(s_full)
    print(f'  x_PCA200: mean={kurt_x.mean():.3f}, |max|={np.abs(kurt_x).max():.3f}, '
          f'frac |k|>1={(np.abs(kurt_x)>1).mean():.3f}')
    print(f'  s_ICA200: mean={kurt_s.mean():.3f}, |max|={np.abs(kurt_s).max():.3f}, '
          f'frac |k|>1={(np.abs(kurt_s)>1).mean():.3f}')
    # ICA objective is non-Gaussianity. Mean |kurt| should be HIGHER for ICA
    print(f'  → ICA 高斯性损失 (mean |kurt_s|={np.abs(kurt_s).mean():.3f} vs |kurt_x|={np.abs(kurt_x).mean():.3f})')

    # Pairwise MI comparison
    print('\n' + '=' * 70)
    print(f'两两成分互信息 (KSG estimator, 抽样 {N_PAIRS_PER_PAIR} 对)')
    print('=' * 70)
    rng = np.random.default_rng(123)
    n_comp = s_full.shape[1]
    pairs = np.array([(i, j) for i in range(n_comp) for j in range(i+1, n_comp)])
    pair_idx = rng.choice(len(pairs), size=min(N_PAIRS_PER_PAIR, len(pairs)), replace=False)
    pairs_sample = pairs[pair_idx]
    print(f'  total pairs: {len(pairs)}, sampled: {len(pairs_sample)}')

    t0 = time.time()
    mi_s = np.empty(len(pairs_sample))
    mi_x = np.empty(len(pairs_sample))
    for k, (i, j) in enumerate(pairs_sample):
        mi_s[k] = fast_mi_bits_pair(s_full[:, i], s_full[:, j], seed=42+k)
        mi_x[k] = fast_mi_bits_pair(x_pca200[:, i], x_pca200[:, j], seed=42+k)
    print(f'  time: {time.time()-t0:.1f}s')

    # Stats
    print(f'\n  MI(x_PCA200)  mean={mi_x.mean():.4f}, median={np.median(mi_x):.4f}, '
          f'p99={np.quantile(mi_x, 0.99):.4f}, max={mi_x.max():.4f}')
    print(f'  MI(s_ICA200)  mean={mi_s.mean():.4f}, median={np.median(mi_s):.4f}, '
          f'p99={np.quantile(mi_s, 0.99):.4f}, max={mi_s.max():.4f}')

    # Paired reduction
    delta = mi_x - mi_s
    n_reduced = (delta > 0.01).sum()
    print(f'\n  ΔMI = MI(x) - MI(s), mean={delta.mean():+.4f}, '
          f'median={np.median(delta):+.4f}, '
          f'frac(Δ>0.01)={n_reduced/len(delta):.3f}')
    # Shuffle control
    delta_shuf = np.empty(len(pairs_sample))
    for k, (i, j) in enumerate(pairs_sample):
        rng2 = np.random.default_rng(999+k)
        s_shuf_i = s_full[rng2.permutation(s_full.shape[0]), i]
        delta_shuf[k] = fast_mi_bits_pair(s_shuf_i, s_full[:, j], seed=42+k)
    print(f'  MI_shuf (sanity ceiling): mean={delta_shuf.mean():.4f}')

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea A 现象 1')
    print('=' * 70)
    # Kill line: ICA MI ≈ original MI (Δ ≤ 0.01 mean)
    kill = '✗ kill (Idea A dead — flan-t5 already near-independent)' if delta.mean() < 0.01 else \
           '✓ ICA 解纠缠有效'
    print(f'  ΔMI mean = {delta.mean():+.4f} bit')
    print(f'  Reduce frac = {n_reduced/len(delta):.3f} (>0.5 算 ICA 显著)')
    print(f'  Verdict: {kill}')

    # Save
    out = {
        'n_ica_components': N_ICA_COMP,
        'n_subsample_ica_fit': N_SUBSAMPLE,
        'n_pairs_sampled': int(len(pairs_sample)),
        'kurtosis_x_PCA200': {'mean_abs': float(np.abs(kurt_x).mean()), 'frac_gt1': float((np.abs(kurt_x)>1).mean())},
        'kurtosis_s_ICA200': {'mean_abs': float(np.abs(kurt_s).mean()), 'frac_gt1': float((np.abs(kurt_s)>1).mean())},
        'mi_x_PCA200_mean': float(mi_x.mean()),
        'mi_x_PCA200_median': float(np.median(mi_x)),
        'mi_s_ICA200_mean': float(mi_s.mean()),
        'mi_s_ICA200_median': float(np.median(mi_s)),
        'delta_mi_mean': float(delta.mean()),
        'delta_mi_median': float(np.median(delta)),
        'delta_mi_p25': float(np.quantile(delta, 0.25)),
        'frac_reduced_gt_0p01': float(n_reduced/len(delta)),
        'kill_verdict': kill,
    }
    with open(os.path.join(OUT_DIR, 'ideaA_phenomenon1.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {OUT_DIR}/ideaA_phenomenon1.json')


if __name__ == '__main__':
    main()