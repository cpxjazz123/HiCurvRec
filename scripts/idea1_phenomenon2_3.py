#!/usr/bin/env python3
"""Idea 1 现象 2 + 3: 注入 CF 后 V-info 回升 + shuffle 对照

Fast MI estimator: sklearn mutual_info_classif (KSG-based, vectorized C).
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json, time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

CF_PATH      = os.path.join(OUT_DIR, 'cf_ppmi_svd256.pt')
EMB_PATH     = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}

N_MI_SAMPLE = 2000      # KSG subsample for speed
ALPHA_GRID = [0.1, 0.3, 0.5]
PROJ_DIM = 64
PCA_RED_DIM = 16       # Reduce residuals to 16-dim for MI (KSG sensitive to dim)


def fast_mi_bits(x, y, n_sample=N_MI_SAMPLE, seed=42, n_neighbors=5):
    """MI in bits via sklearn mutual_info_classif (KSG-based, fast).

    x: (N, D), y: (N,) categorical
    Returns scalar MI in bits.

    KSG estimator suffers in high dim. Use PCA(PCA_RED_DIM) reduction first if D > 32.
    """
    if x.ndim == 2 and x.shape[1] > 32:
        from sklearn.decomposition import PCA as _PCA
        x = _PCA(n_components=PCA_RED_DIM, random_state=seed).fit_transform(x)
    N = x.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    # mutual_info_classif returns log-base-e MI; convert to bits
    mi_nats = mutual_info_classif(
        x_s, y_s,
        n_neighbors=n_neighbors,
        random_state=seed,
        discrete_features=False,
        n_jobs=4,
    )
    mi_bits_per_feat = mi_nats.mean() / np.log(2)
    return float(mi_bits_per_feat)


def build_y(emb_path, K=20):
    x = torch.load(emb_path, map_location='cpu', weights_only=False).float()
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024).fit(x.numpy())
    return km.labels_.astype(np.int64)


def fit_proj(r_l_np, f_np, dim=PROJ_DIM, seed=42):
    """Fit A: f (256-dim) → PCA(r_l, dim=64)-lifted basis.
    Returns A (256, dim) and pca (so we can lift to D later).
    """
    pca = PCA(n_components=dim, random_state=seed)
    r_reduced = pca.fit_transform(r_l_np)
    A, *_ = np.linalg.lstsq(f_np, r_reduced, rcond=None)
    return A.astype(np.float32), pca


def main():
    print('=' * 70)
    print('Loading PPMI f_i, embedding, Y proxy')
    print('=' * 70)
    cf = torch.load(CF_PATH, map_location='cpu', weights_only=False)
    f = cf['f_normalized'].numpy().astype(np.float32)
    print(f'  f: {f.shape}')
    y = build_y(EMB_PATH, K=20)
    print(f'  y: {y.shape}, distribution={np.bincount(y).tolist()}')

    rng = np.random.default_rng(123)
    f_shuffled = f[rng.permutation(f.shape[0])]

    print('\nLoading r_lst for A/B/C')
    bundles = {n: torch.load(p, map_location='cpu', weights_only=False)
               for n, p in RQIDX_PATHS.items()}

    print('\nBaseline V-info: I_V(r_l → Y)')
    base_vinfo = {}
    for name, bundle in bundles.items():
        r_lst = bundle['r_lst']
        base_vinfo[name] = {}
        for l in range(1, len(r_lst)):
            r = r_lst[l].numpy().astype(np.float32)
            mi = fast_mi_bits(r, y)
            base_vinfo[name][f'l{l}'] = mi
            print(f'  {name} L{l}: I_V(r_l → Y) = {mi:.4f} bit')

    mi_cf = fast_mi_bits(f, y)
    print(f'\nCF-only: I_V(f → Y) = {mi_cf:.4f} bit')

    print('\n' + '=' * 70)
    print('Per algo × layer: inject α·Proj(f) and compute V-info')
    print('=' * 70)
    results = {
        'alpha_sweep': {},
        'baseline_vinfo': base_vinfo,
        'cf_only_vinfo': mi_cf,
        'y_kmeans_K': 20,
        'proj_dim': PROJ_DIM,
        'alpha_grid': ALPHA_GRID,
        'n_mi_sample': N_MI_SAMPLE,
    }

    for name, bundle in bundles.items():
        print(f'\n=== {name} ===')
        r_lst = bundle['r_lst']
        results['alpha_sweep'][name] = {}
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            r_scale = float(np.linalg.norm(r_l, axis=-1).mean())
            A, pca = fit_proj(r_l, f, dim=PROJ_DIM, seed=42)
            # proj_lifted: (N, D) — full residual-dim projection of f
            fA = (f @ A)                                # (N, dim)
            proj_lifted = fA @ pca.components_           # (N, D)
            proj_scale = float(np.linalg.norm(proj_lifted, axis=-1).mean())
            print(f'  L{l}: ‖r_l‖={r_scale:.4f}, ‖Proj(f)‖={proj_scale:.4f}, '
                  f'ratio={proj_scale/r_scale:.4f}')
            results['alpha_sweep'][name][f'l{l}'] = {
                'r_scale': r_scale,
                'proj_scale': proj_scale,
            }
            for alpha in ALPHA_GRID:
                t0 = time.time()
                r_tilde = r_l + alpha * proj_lifted
                r_tilde_shuf = r_l + alpha * ((f_shuffled @ A) @ pca.components_)
                mi_inj = fast_mi_bits(r_tilde, y, seed=42)
                mi_shuf = fast_mi_bits(r_tilde_shuf, y, seed=42)
                dt = time.time() - t0
                d_inj = mi_inj - base_vinfo[name][f'l{l}']
                d_shuf = mi_shuf - base_vinfo[name][f'l{l}']
                results['alpha_sweep'][name][f'l{l}'][f'alpha{alpha}'] = {
                    'v_info_inject':    mi_inj,
                    'delta_v_inject':   d_inj,
                    'v_info_shuffled':  mi_shuf,
                    'delta_v_shuffled': d_shuf,
                    'compute_time_s':   dt,
                }
                print(f'    α={alpha}: I_V(r̃→Y)={mi_inj:.4f}, Δ={d_inj:+.4f}, '
                      f'shuf_Δ={d_shuf:+.4f}  ({dt:.1f}s)')

    out_path = os.path.join(OUT_DIR, 'idea1_phenomenon2_3.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n=== Saved → {out_path} ===')

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea 1 现象 2 + 3')
    print('=' * 70)
    for name in results['alpha_sweep']:
        for lk in sorted(results['alpha_sweep'][name].keys()):
            best_alpha = max(ALPHA_GRID, key=lambda a: results['alpha_sweep'][name][lk][f'alpha{a}']['delta_v_inject'])
            d = results['alpha_sweep'][name][lk][f'alpha{best_alpha}']['delta_v_inject']
            ds = results['alpha_sweep'][name][lk][f'alpha{best_alpha}']['delta_v_shuffled']
            gap = d - ds
            kill2 = '✓ >0.1' if d > 0.1 else '✗ <0.1'
            kill3 = '✓ shuf<<inj' if abs(ds) < 0.5 * abs(d) else '✗ shuf≈inj (leakage)'
            print(f'  {name} {lk} α={best_alpha}: ΔI_V(inj)={d:+.4f}, ΔI_V(shuf)={ds:+.4f}, '
                  f'gap={gap:+.4f}  {kill2} {kill3}')


if __name__ == '__main__':
    main()