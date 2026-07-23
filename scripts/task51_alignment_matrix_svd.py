#!/usr/bin/env python3
"""Task 309 v2: 用真实 SVD b_beh 升级对齐矩阵

利用 cooccurrence_csr.npz (11924×11924, 1.88M nnz) 算 item-item SVD-50 作为 b_beh,
重新跑 8 targets × 4 metrics 对齐矩阵.

8 targets:
- x_sem (FLAN-T5 2048-d)
- r_sem_L1/L2/L3 (semantic residual)
- b_beh (item-item SVD-50, 真实协同 embedding)
- b_beh_residual (FLAN-T5 - b_beh projection)
- pop (popularity proxy)

4 metrics: R² (linear probe), CKA, NN-Overlap@10, POP
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task51_alignment_svd'
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_PT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
COOCC = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate/cooccurrence_csr.npz'

PCA_DIM = 30
SVD_DIM = 50


def linear_probe_r2_multidim(X_feat, Y_target, alpha=1.0):
    X_train, X_test, Y_train, Y_test = train_test_split(X_feat, Y_target, test_size=0.2, random_state=42)
    m = Ridge(alpha=alpha)
    m.fit(X_train, Y_train)
    Y_pred = m.predict(X_test)
    r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')
    return float(np.mean(r2))


def cka(X, Y):
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = Xc.T @ Yc
    num = (cross ** 2).sum()
    xx = Xc.T @ Xc
    yy = Yc.T @ Yc
    den = np.sqrt(((xx ** 2).sum()) * ((yy ** 2).sum()) + 1e-12)
    return float(num / (den + 1e-12))


def nn_overlap(X, Y, k=10):
    N = X.shape[0]
    nn_x = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    nn_y = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Y)
    nx = nn_x.kneighbors(X, return_distance=False)[:, 1:k + 1]
    ny = nn_y.kneighbors(Y, return_distance=False)[:, 1:k + 1]
    overlap = np.zeros(N)
    for i in range(N):
        overlap[i] = len(set(nx[i]) & set(ny[i])) / k
    return float(overlap.mean())


def pop_score(X, Y, sample=2000, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    i_idx = rng.integers(0, N, size=sample)
    j_idx = rng.integers(0, N, size=sample)
    k_idx = rng.integers(0, N, size=sample)
    d_x = np.linalg.norm(X[i_idx] - X[j_idx], axis=-1) - np.linalg.norm(X[i_idx] - X[k_idx], axis=-1)
    d_y = np.linalg.norm(Y[i_idx] - Y[j_idx], axis=-1) - np.linalg.norm(Y[i_idx] - Y[k_idx], axis=-1)
    agree = ((d_x < 0) == (d_y < 0)).mean()
    return float(agree)


def main():
    print('=' * 70)
    print('Task 309 v2: 对齐矩阵 (真实 SVD b_beh 升级版)')
    print('=' * 70)

    # Load semantic embedding
    print('\n[1] 加载 FLAN-T5 embedding...')
    x_sem = torch.load(EMBED_PT, map_location='cpu', weights_only=False)
    if x_sem.dim() > 2:
        x_sem = x_sem.squeeze(0)
    if x_sem.shape[0] != 11924 and x_sem.shape[1] == 11924:
        x_sem = x_sem.t()
    N = x_sem.shape[0]
    x_sem_np = x_sem.float().numpy()
    print(f'  x_sem shape: {x_sem_np.shape}, N={N}')

    # Load cooccurrence matrix
    print(f'\n[2] 加载 cooccurrence matrix...')
    cooc = load_npz(COOCC)
    print(f'  cooc shape: {cooc.shape}, nnz: {cooc.nnz}')

    # SVD-50 for behavioral embedding
    print(f'\n[3] SVD-{SVD_DIM} on cooccurrence → b_beh...')
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    b_beh = svd.fit_transform(cooc).astype(np.float32)  # (N, 50)
    print(f'  b_beh shape: {b_beh.shape}, var_explained sum: {svd.explained_variance_ratio_.sum():.3f}')
    # Save SVD variance explained per dim
    svd_var = svd.explained_variance_ratio_

    # Load residual bundle
    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    r_lst = bundle['r_lst']
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    L = len(codebooks)

    # Compute q^l (cumulative)
    q_per_layer = []
    q_cum = np.zeros((N, x_sem_np.shape[1]), dtype=np.float32)
    for l in range(3):
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum += q_l
        q_per_layer.append(q_cum.copy())

    # Compute semantic residuals: r^l = x - q_<l
    r_alt_layers = []
    q_cum_alt = np.zeros((N, x_sem_np.shape[1]), dtype=np.float32)
    for l in range(3):
        r_alt = x_sem_np - q_cum_alt
        r_alt_layers.append(r_alt.astype(np.float32))
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum_alt += q_l

    # Behavioral residual: project x_sem to b_beh space, then x - b_beh projection
    # Use linear projection x_sem @ V where V is from b_beh = cooc @ V
    print(f'\n[4] 构造 behavioral residual (x - b_beh projection)...')
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_sem_np, b_beh)
    b_beh_proj = ridge.predict(x_sem_np)  # (N, SVD_DIM)
    # ridge.coef_ shape: (SVD_DIM, 2048), so b_beh_proj @ ridge.coef_ has shape (N, 2048)
    b_beh_recon = b_beh_proj @ ridge.coef_  # (N, 2048)
    r_beh = (x_sem_np - b_beh_recon).astype(np.float32)  # residual in x_sem space
    # Project to SVD space for direct comparison
    pca_r_beh = PCA(n_components=SVD_DIM).fit_transform(r_beh).astype(np.float32)

    # Popularity proxy: row sum of cooccurrence
    pop_target = np.array(cooc.sum(axis=1)).flatten().astype(np.float32)
    print(f'  pop_target: min={pop_target.min()}, max={pop_target.max()}, mean={pop_target.mean():.1f}')

    # Build target dict (8 targets as per spec)
    targets = {
        'x_sem': x_sem_np.astype(np.float32),
        'r_sem_L1': r_alt_layers[0],
        'r_sem_L2': r_alt_layers[1],
        'r_sem_L3': r_alt_layers[2],
        'b_beh': b_beh,
        'r_beh': pca_r_beh,
        'pop': pop_target.reshape(-1, 1),
    }

    # Per-layer h^(l) = q^(l) (cumulative codebook)
    h_per_layer = []
    for l in range(3):
        h_per_layer.append(q_per_layer[l])

    # Compute alignment
    print('\n[5] 计算 8 targets × 4 metrics...')
    alignment = {}
    for l in range(3):
        layer_name = f'L{l+1}'
        print(f'\n  [{layer_name}] alignment:')
        h_l = h_per_layer[l]
        # Reduce h_l dim
        pca_h = PCA(n_components=PCA_DIM, random_state=42).fit_transform(h_l)
        for t_name, t_vec in targets.items():
            print(f'    {t_name}: ', end='', flush=True)
            # Reduce target dim
            if t_vec.ndim == 1:
                t_use = t_vec.reshape(-1, 1)
            elif t_vec.shape[1] > PCA_DIM:
                t_use = PCA(n_components=PCA_DIM, random_state=42).fit_transform(t_vec)
            else:
                t_use = t_vec
            r2 = linear_probe_r2_multidim(pca_h, t_use)
            cka_v = cka(pca_h, t_use)
            nn_o = nn_overlap(pca_h, t_use, k=10)
            pop_v = pop_score(pca_h, t_use, sample=1000, seed=42)
            alignment[(layer_name, t_name)] = {
                'r2': r2,
                'cka': cka_v,
                'nn_overlap_10': nn_o,
                'pop': pop_v,
            }
            print(f'R²={r2:.3f}, CKA={cka_v:.3f}, NN={nn_o:.3f}, POP={pop_v:.3f}')

    # Build matrix
    matrix = {}
    for l in range(3):
        layer_name = f'L{l+1}'
        matrix[layer_name] = {t: alignment[(layer_name, t)] for t in targets.keys()}

    # Save
    out = {
        'n_items': N,
        'embed_path': EMBED_PT,
        'coocc_path': COOCC,
        'svd_dim': SVD_DIM,
        'svd_var_explained_sum': float(svd_var.sum()),
        'svd_var_per_dim': svd_var.tolist(),
        'note': 'b_beh = SVD-50(cooccurrence_csr). 真实协同 embedding.',
        'targets': list(targets.keys()),
        'matrix': matrix,
    }
    with open(os.path.join(OUT_DIR, 'alignment_matrix.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/alignment_matrix.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 309 v2 Verdict: 对齐矩阵 (真实 SVD b_beh)\n\n')
        f.write(f'数据集: Toys (N={N}), SVD-{SVD_DIM} on cooccurrence (var_explained={svd_var.sum():.3f})\n\n')
        f.write('## 8 Targets\n\n')
        f.write('- `x_sem`: FLAN-T5 2048-d\n')
        f.write('- `r_sem_L1/L2/L3`: 累计残差 (x - Σ_{j<l} q^j)\n')
        f.write('- `b_beh`: SVD-50 of cooccurrence_csr.npz (1.88M nnz)\n')
        f.write('- `r_beh`: x - b_beh projection (PCA-50)\n')
        f.write('- `pop`: cooccurrence row sum\n\n')
        f.write('## 对齐矩阵 (3 layers × 7 targets)\n\n')
        for l in range(3):
            layer_name = f'L{l+1}'
            f.write(f'### {layer_name}\n\n')
            f.write('| Target | R² | CKA | NN@10 | POP |\n')
            f.write('|--------|-----|-----|-------|-----|\n')
            for t_name in targets.keys():
                e = matrix[layer_name][t_name]
                f.write(f'| {t_name} | {e["r2"]:.3f} | {e["cka"]:.3f} | '
                        f'{e["nn_overlap_10"]:.3f} | {e["pop"]:.3f} |\n')
            f.write('\n')

        f.write('\n## 关键判读\n\n')
        for l in range(3):
            layer_name = f'L{l+1}'
            sem_r2 = matrix[layer_name][f'r_sem_L{l+1}']['r2']
            beh_r2 = matrix[layer_name]['b_beh']['r2']
            ratio = sem_r2 / (beh_r2 + 1e-6)
            f.write(f'### {layer_name}\n')
            f.write(f'- R²(r_sem) / R²(b_beh) = {sem_r2:.3f} / {beh_r2:.3f} = {ratio:.2f}\n')
            if ratio > 1.5:
                f.write(f'  → **语义主导**\n')
            elif ratio < 0.7:
                f.write(f'  → **行为 (协同) 主导**\n')
            else:
                f.write(f'  → **混合编码**\n')
            f.write('\n')

        # Compare with v1 (q_total proxy)
        f.write('\n## 与 v1 (累计 codebook proxy) 对比\n\n')
        f.write('| Layer | v1 R²(b_beh) | v2 R²(b_beh real) | Δ |\n')
        f.write('|-------|---------------|---------------------|---|\n')
        v1_b_beh_r2 = {'L1': 0.750, 'L2': 0.910, 'L3': 0.984}  # from v1 result
        for l in range(3):
            layer_name = f'L{l+1}'
            v2_r2 = matrix[layer_name]['b_beh']['r2']
            v1_r2 = v1_b_beh_r2[layer_name]
            delta = v2_r2 - v1_r2
            f.write(f'| {layer_name} | {v1_r2:.3f} | {v2_r2:.3f} | {delta:+.3f} |\n')
        f.write('\n注: v1 b_beh 是累计 codebook (构造性等于 h^(l)), v2 b_beh 是真实协同 SVD.\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()