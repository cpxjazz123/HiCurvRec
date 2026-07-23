#!/usr/bin/env python3
# task17_info_tok_plane.py — task19 #3 InfoTok 信息平面
#
# 12 个小 classifier per layer × 3 algorithms:
#   - 输入: per-layer SID codes c_l (label: y = item category proxy via Group A embedding cluster)
#   - 测 I(Z;Y): mutual information Z(l) -> Y
#   - 测 I(Z;X): mutual information Z(l) -> X
#   - 用 NCE-style estimator (Kraskov + sklearn KNN)
#
# 输出: I(Z;Y) and I(Z;X) 平面, per algorithm × layer
#
# CPU only, ~30 min on 11924 items × 8 layers

import sys, os, json, time
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch
from scipy.spatial import cKDTree

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task19'
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_A = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
GROUP_B = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt'
GROUP_C = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt'


def kraskov_mi(x, y, k=5, n_subsample=2000):
    """Kraskov KSG-1 mutual information estimator (MI in bits).

    x: (N, d_x), y: (N, d_y), typically d_y=1 for label
    Uses cKDTree (Chebyshev / max-norm) to support per-point radii.
    """
    N = x.shape[0]
    rng = np.random.default_rng(42)
    if N > n_subsample:
        idx = rng.choice(N, size=n_subsample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    N_s = x_s.shape[0]
    if y_s.ndim == 1:
        y_s = y_s.reshape(-1, 1)

    # Joint kNN: Chebyshev (max-norm), supports p=inf
    xy = np.concatenate([x_s, y_s], axis=1)
    tree_xy = cKDTree(xy)
    # (k+1)th neighbor distance (excluding self); kneighbors returns (distances, indices)
    dists, _ = tree_xy.query(xy, k=k+1, p=np.inf)
    if k == 1:
        eps_x = dists[:, k]
    else:
        eps_x = dists[:, k]

    # Marginal X: count within eps_x in X-only Chebyshev
    tree_x = cKDTree(x_s)
    # cKDTree.query_ball_point supports array of radii via single calls
    nx_count = np.empty(N_s, dtype=np.int64)
    for i in range(N_s):
        idxs = tree_x.query_ball_point(x_s[i], r=eps_x[i], p=np.inf)
        nx_count[i] = len(idxs) - 1   # -1 excludes self (query point is in its own ball)
    tree_y = cKDTree(y_s)
    ny_count = np.empty(N_s, dtype=np.int64)
    for i in range(N_s):
        idxs = tree_y.query_ball_point(y_s[i], r=eps_x[i], p=np.inf)
        ny_count[i] = len(idxs) - 1

    from scipy.special import digamma
    mi = digamma(k) + digamma(N_s) - np.mean(digamma(nx_count + 1) + digamma(ny_count + 1))
    mi_bits = mi / np.log(2)
    return float(mi_bits)


def main():
    print('=' * 70)
    print('task19 #3 InfoTok — I(Z;Y) and I(Z;X) per algorithm × layer')
    print('=' * 70)

    algorithms = {
        'A_baseline': GROUP_A,
        'B_mmq': GROUP_B,
        'C_gsrq': GROUP_C,
    }

    # Y target: per-item "category" proxy = cluster ID on raw embedding (using HKMeans on x)
    # 我们用 flan-t5-xl embedding 本身做 Y latent; 简化: 用 RQ L1 code 当粗 label
    # 用 Group A L1 code 当 y (但这会与 Z 强相关; 改用 KMeans on x with K=20)
    print('Computing Y-label via KMeans(K=20) on x...')
    x_a = torch.load(GROUP_A, weights_only=False, map_location='cpu')['r_lst'][0].float()
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3, batch_size=1024).fit(x_a.numpy())
    y_label = kmeans.labels_.astype(np.int64)
    print(f'  y_label.shape={y_label.shape}, distribution={np.bincount(y_label)}')

    results = {}
    for name, path in algorithms.items():
        print(f'\n=== {name} ===')
        bundle = torch.load(path, weights_only=False, map_location='cpu')
        idx_lst = bundle['idx_lst']
        L = len(idx_lst)

        # Build Z(l) one-hot per layer
        z_per_layer = []
        for l in range(L):
            codes = idx_lst[l].numpy()
            K = codes.max() + 1
            z = np.zeros((codes.shape[0], K), dtype=np.float32)
            z[np.arange(codes.shape[0]), codes] = 1.0
            # SVD to 16-d for MI estimation (KSG sensitive to dim)
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=16, random_state=42)
            z_red = svd.fit_transform(z)
            z_per_layer.append(z_red)
            print(f'  L{l+1}: codes shape {codes.shape}, Z_red {z_red.shape}')

        # Compute I(Z;Y) and I(Z;X) per layer
        info = {'per_layer': {}}
        for l in range(L):
            z = z_per_layer[l].astype(np.float32)
            t0 = time.time()
            mi_zy = kraskov_mi(z, y_label, k=5, n_subsample=1500)
            # For I(Z;X) we use idx_lst[l] (the raw code) since Z = codes + a few orthogonal directions
            # Use r_l (residual) as X signal (since codes alone can't have MI with X)
            r_l = bundle['r_lst'][l+1].float().numpy()       # residual at layer l
            from sklearn.decomposition import TruncatedSVD
            svd2 = TruncatedSVD(n_components=16, random_state=42)
            r_red = svd2.fit_transform(r_l.astype(np.float32))
            mi_zx = kraskov_mi(z, r_red, k=5, n_subsample=1500)

            elapsed = time.time() - t0
            info['per_layer'][l] = {
                'mi_zy_bits': mi_zy,
                'mi_zx_residual_bits': mi_zx,
                'time_sec': elapsed,
            }
            print(f'  L{l+1}: I(Z;Y)={mi_zy:.3f} bits, I(Z;X|r)={mi_zx:.3f} bits ({elapsed:.0f}s)')

        # Aggregate per-layer means
        mi_zy_seq = [info['per_layer'][l]['mi_zy_bits'] for l in range(L)]
        mi_zx_seq = [info['per_layer'][l]['mi_zx_residual_bits'] for l in range(L)]
        info['mi_zy_sequence'] = mi_zy_seq
        info['mi_zx_residual_sequence'] = mi_zx_seq

        # Kill-line check
        delta_zy = mi_zy_seq[-1] - mi_zy_seq[0]
        delta_zx = mi_zx_seq[-1] - mi_zx_seq[0]
        info['delta_zy_Lv_from_L1'] = delta_zy
        info['delta_zx_Lv_from_L1'] = delta_zx
        info['kill_lines'] = {
            'task17_3_zy_should_keep_growing': delta_zy > 0.1,
            'task17_3_zx_residual_should_decrease': delta_zx < 0,
        }
        results[name] = info
        print(f'  I(Z;Y) Δ L{l}→L1={delta_zy:.3f} bits, I(Z;X|r) Δ={delta_zx:.3f}')

    # Save
    out_path = os.path.join(OUT_DIR, 'task17_info_tok_plane.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n=== JSON → {out_path} ===')

    # Summary MD
    summary = ['# task19 #3 InfoTok 信息平面 findings', '']
    summary.append('| Algorithm | Layer | I(Z;Y) bits | I(Z;X\\|r) bits |')
    summary.append('|-----------|-------|-------------|-----------------|')
    for name in algorithms:
        info = results[name]
        for l in range(L):
            v = info['per_layer'][l]
            summary.append(f'| {name} | L{l+1} | {v["mi_zy_bits"]:.3f} | {v["mi_zx_residual_bits"]:.3f} |')
    summary.append('')
    summary.append('## 序贯')
    for name in algorithms:
        info = results[name]
        dzy = info['delta_zy_Lv_from_L1']
        dzx = info['delta_zx_Lv_from_L1']
        summary.append(f'- **{name}**: Δ I(Z;Y) L{l}→L1 = {dzy:.3f} bits, Δ I(Z;X|r) L{l}→L1 = {dzx:.3f} bits')

    summary.append('')
    summary.append('## kill line')
    summary.append('- task17_3: 深层 I(Z;Y) 应仍在涨 (Δ > 0.1 bit) → 否则 "non-minimal SID" 假设被否')
    summary.append('- task17_3: 深层 I(Z;X|r) 应在降 (Δ < 0) → 否则 残差越深越信息丰富, 不符理论预期')

    md_path = os.path.join(OUT_DIR, 'task17_info_tok_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f'\n=== Summary → {md_path} ===')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
