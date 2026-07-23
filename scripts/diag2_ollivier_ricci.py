#!/usr/bin/env python3
"""HRQ 诊断2: Ollivier-Ricci 曲率 — 局部曲率估计, 比 Gromov δ 更稳健

目的: 用 k-NN 图 + W1 Wasserstein 估计每条边的局部曲率 κ_OR

公式:
  κ_OR(i, j) = 1 - W1(m_i, m_j) / d(x_i, x_j)
  其中 m_i = uniform over k-NN(i), W1 用 LLY 近似:
    W1 ≈ (1/k) * Σ_{a∈N(i)} min_{b∈N(j)} d(a, b)

判定:
- κ_OR < 0 → 局部负曲率 (树状分叉)
- κ_OR > 0 → 局部正曲率 (团簇)
- κ_OR ≈ 0 → 平坦
- 分布显著偏负 → 数据局部确实有分叉倾向 (Gromov δ 和持续同调没测出来)

kill 线: κ_OR 集中在 0 附近或偏正 → 局部无分叉证据, 进一步坐实"数据本身不是树状"

注: 我们这里只用 X_raw 做诊断 (因为 HRQ 是在 X_raw 上的"加工",
   HRQ 编码器的局部曲率应该在原始空间就有体现, 否则就没动机.
   如果 X_raw 没有局部负曲率, HRQ 没有理由有.)

复用:
- Stage 1 flan-t5-xl embedding
- PCA-50 白化 (匹配 idea5 的几何尺度)
"""
import sys
sys.path.insert(0, '/home/wlia0047/arenyu/GeneRec/GRID' if False else '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa

import os, json
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag2_ollivier_ricci'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

N_SUBSAMPLE = 2000  # item 子采样
K_NN = 10           # k-nn neighbors
PCA_DIM = 50        # 匹配 idea5 协议
SEED = 42


def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    return (X_centered @ V) / (S[:n_components].clamp(min=1e-8))


def compute_kappa_OR_for_edge(x_i, x_j, N_i, N_j, k):
    """LLY approximation: W1(m_i, m_j) ≈ (1/k) Σ_{a∈N(i)} min_{b∈N(j)} d(a, b)
       κ_OR = 1 - W1 / d(x_i, x_j)
       Returns: κ_OR (or NaN if degenerate)
    """
    x_i = x_i.reshape(1, -1)
    x_j = x_j.reshape(1, -1)
    d_ij = float(np.linalg.norm(x_i - x_j))
    if d_ij < 1e-12:
        return float('nan')

    # Compute d(a, N(j)) for each a ∈ N(i)
    # (k, d) vs (k, d) -> (k, k) distance matrix
    D = np.linalg.norm(N_i[:, None] - N_j[None], axis=-1)
    min_d = D.min(axis=1)  # (k,) — min over j_neighbors for each i_neighbor
    W1 = float(min_d.mean())
    return 1.0 - W1 / d_ij


def main():
    print('=' * 70)
    print('HRQ 诊断2: Ollivier-Ricci 曲率')
    print('=' * 70)

    print('\n[Step 1] 加载 Stage 1 embedding, 子采样 + PCA 白化')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N_full, D_eu = x_eu.shape
    print(f'  Embedding: {tuple(x_eu.shape)}')

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(N_full, size=min(N_SUBSAMPLE, N_full), replace=False)
    x_sub = x_eu[sub_idx]
    print(f'  Subsampled: {x_sub.shape}')

    x_white = whiten_pca(x_sub, n_components=PCA_DIM).numpy()
    print(f'  After PCA-{PCA_DIM}: {x_white.shape}')

    print(f'\n[Step 2] 构建 k={K_NN} NN 图')
    t0 = __import__('time').time()
    knn = NearestNeighbors(n_neighbors=K_NN + 1, algorithm='auto', n_jobs=-1)
    knn.fit(x_white)
    distances, indices = knn.kneighbors(x_white)
    print(f'  kNN time: {__import__("time").time() - t0:.2f}s')

    # 移除 self (distance 0 to self)
    # indices[i, 0] should be i itself; skip if so
    if (indices[:, 0] == np.arange(len(indices))).all():
        nbrs = indices[:, 1:]  # (N, K_NN) — actual neighbors
        nbr_dists = distances[:, 1:]
    else:
        nbrs = indices[:, :K_NN]
        nbr_dists = distances[:, :K_NN]
    print(f'  Neighbors array: {nbrs.shape}')

    print(f'\n[Step 3] 计算 κ_OR for each edge')
    kappa_values = []
    edge_count = 0
    for i in range(len(x_white)):
        for kk in range(K_NN):
            j = int(nbrs[i, kk])
            if i == j:
                continue
            # 取对称邻居 (避免重复)
            if j < i:
                continue
            N_i = x_white[nbrs[i]]  # (K_NN, d)
            N_j = x_white[nbrs[j]]  # (K_NN, d)
            kappa = compute_kappa_OR_for_edge(
                x_white[i], x_white[j], N_i, N_j, K_NN
            )
            if not np.isnan(kappa):
                kappa_values.append(kappa)
            edge_count += 1
        if i % 200 == 0:
            print(f'    center {i}/{len(x_white)}, {len(kappa_values)} edges so far', flush=True)

    kappa_arr = np.array(kappa_values)
    print(f'\n[Step 4] κ_OR 分布:')
    print(f'  n_edges = {len(kappa_arr)}')
    print(f'  mean = {kappa_arr.mean():.4f}')
    print(f'  std  = {kappa_arr.std():.4f}')
    print(f'  q25  = {np.percentile(kappa_arr, 25):.4f}')
    print(f'  q50  = {np.percentile(kappa_arr, 50):.4f}')
    print(f'  q75  = {np.percentile(kappa_arr, 75):.4f}')
    print(f'  frac_neg = {(kappa_arr < 0).mean():.4f}')
    print(f'  frac_pos = {(kappa_arr > 0).mean():.4f}')

    # 判定
    # Standard heuristic: 在树上 κ_OR 应该在 [-0.4, -0.6] 之间
    # 在欧氏网格上 κ_OR ≈ 0
    # 在正曲率空间 (sphere) 上 κ_OR > 0
    mean_kappa = float(kappa_arr.mean())
    frac_neg = float((kappa_arr < 0).mean())
    q50 = float(np.percentile(kappa_arr, 50))

    if mean_kappa < -0.2 and frac_neg > 0.7:
        verdict = (
            f'STRONG_LOCAL_NEG_CURVATURE: mean κ_OR = {mean_kappa:.3f}, '
            f'frac_neg = {frac_neg:.3f}. '
            f'数据局部确实有分叉倾向, 之前 Gromov δ 没测出来. '
            f'HRQ 几何动机 (局部树状) 有局部曲率证据支持.'
        )
        outcome = 'confirmed_local_neg'
    elif mean_kappa < -0.05 and frac_neg > 0.55:
        verdict = (
            f'WEAK_LOCAL_NEG_CURVATURE: mean κ_OR = {mean_kappa:.3f}, '
            f'frac_neg = {frac_neg:.3f}. '
            f'有弱信号但不到"显著"水平. 局部有轻微分叉倾向.'
        )
        outcome = 'weak_local_neg'
    elif abs(mean_kappa) <= 0.05:
        verdict = (
            f'KILL_LOCAL_FLAT: mean κ_OR = {mean_kappa:.3f} (≈ 0). '
            f'数据局部是平坦的, 没有局部树状分叉证据. '
            f'坐实"数据本身不是树状", HRQ +37% 不是来自"匹配树状"假设. '
            f'应转向诊断3 (曲率剂量-反应) 或诊断4 (流行度-距离).'
        )
        outcome = 'local_flat'
    else:
        verdict = (
            f'POSITIVE_CURVATURE_SURPRISE: mean κ_OR = {mean_kappa:.3f} (> 0). '
            f'数据局部呈"团簇状"而非"树状". 这是反 Gromov 假设的发现. '
            f'需要进一步验证 (可能是 PCA 白化引入的几何伪影).'
        )
        outcome = 'local_pos_surprise'

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'ollivier_ricci.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'ollivier_ricci_curvature',
            'description': (
                'LLY approximation: κ_OR(i,j) = 1 - W1(m_i, m_j) / d(x_i, x_j). '
                'W1 via LLY: (1/k) Σ min_{b∈N(j)} d(a, b).'
            ),
            'params': {
                'n_subsample': N_SUBSAMPLE,
                'k_nn': K_NN,
                'pca_dim': PCA_DIM,
                'seed': SEED,
            },
            'kappa_stats': {
                'n_edges': int(len(kappa_arr)),
                'mean': mean_kappa,
                'std': float(kappa_arr.std()),
                'min': float(kappa_arr.min()),
                'max': float(kappa_arr.max()),
                'q25': float(np.percentile(kappa_arr, 25)),
                'q50': q50,
                'q75': float(np.percentile(kappa_arr, 75)),
                'frac_neg': frac_neg,
                'frac_pos': float((kappa_arr > 0).mean()),
            },
            'verdict': verdict,
            'outcome': outcome,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # 画 histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(kappa_arr, bins=80, edgecolor='k', alpha=0.7)
        ax.axvline(mean_kappa, color='red', linestyle='--', label=f'mean={mean_kappa:.3f}')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Ollivier-Ricci curvature κ_OR')
        ax.set_ylabel('count')
        ax.set_title(f'κ_OR distribution (X_raw, k={K_NN})\n'
                     f'mean={mean_kappa:.3f}, std={kappa_arr.std():.3f}, '
                     f'frac_neg={frac_neg:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, 'kappa_or_histogram.png')
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] plot failed: {e}')

    return verdict, kappa_arr


if __name__ == '__main__':
    main()
