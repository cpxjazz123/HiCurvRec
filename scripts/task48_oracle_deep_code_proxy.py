#!/usr/bin/env python3
"""Task 306: Oracle 深层 Code 上限 (proxy 版, 无 TIGER)

5 种 oracle 构造, 比较每种构造的:
- intra-code cos sim (高 = code 内部语义一致)
- coverage (使用 codebook 比例)
- 与 RQ 相比的相对压缩质量

Proxy: 用 FLAN-T5 嵌入空间的 intra-code 一致性作为 oracle gain proxy.
完整版需 TIGER 端到端评估 NDCG.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task48_oracle'
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_PT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def cos_sim(x1, x2):
    a = x1 / (np.linalg.norm(x1, axis=-1, keepdims=True) + 1e-8)
    b = x2 / (np.linalg.norm(x2, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())


def intra_code_cos(X, codes, K, sample_codes=20):
    """For each code, compute mean cos sim of items sharing that code. Return overall mean."""
    sims = []
    rng = np.random.default_rng(42)
    used_codes = [c for c in range(K) if (codes == c).sum() >= 2]
    if not used_codes:
        return 0.0
    sample = rng.choice(used_codes, size=min(sample_codes, len(used_codes)), replace=False)
    for c in sample:
        idx = np.where(codes == c)[0]
        if len(idx) < 2:
            continue
        sub = X[idx]
        # Compute pairwise cos sim
        sub_norm = sub / (np.linalg.norm(sub, axis=-1, keepdims=True) + 1e-8)
        cs = sub_norm @ sub_norm.T
        # Upper triangle mean
        iu = np.triu_indices(len(idx), k=1)
        sims.append(float(cs[iu].mean()))
    return float(np.mean(sims)) if sims else 0.0


def codebook_coverage(codes, K):
    return float((np.bincount(codes, minlength=K) > 0).sum() / K)


def main():
    print('=' * 70)
    print('Task 306: Oracle 深层 Code 上限 (proxy 版)')
    print('=' * 70)

    # Load data
    print('\n[1] 加载 FLAN-T5 embedding...')
    x_sem = torch.load(EMBED_PT, map_location='cpu', weights_only=False)
    if x_sem.dim() > 2:
        x_sem = x_sem.squeeze(0)
    if x_sem.shape[0] != 11924 and x_sem.shape[1] == 11924:
        x_sem = x_sem.t()
    N = x_sem.shape[0]
    X = x_sem.float().numpy()
    print(f'  X shape: {X.shape}')

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    r_lst = bundle['r_lst']
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    L = len(codebooks)
    K_per_layer = [codebooks[l].shape[0] for l in range(L)]
    print(f'  K per layer: {K_per_layer}')

    # Dimensionality reduce for K-means speed
    from sklearn.decomposition import PCA
    print('\n[2] PCA-100 for K-means...')
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f'  X_pca: {X_pca.shape}')

    # RQ baseline per-layer
    print('\n[3] RQ baseline (L1/L2/L3)...')
    rq_results = {}
    for l in range(3):
        codes = idx_lst[l].numpy().astype(int)
        K = K_per_layer[l]
        rq_results[f'L{l+1}'] = {
            'intra_cos': intra_code_cos(X, codes, K),
            'coverage': codebook_coverage(codes, K),
            'method': 'RQ_baseline',
        }
        print(f'  L{l+1} (K={K}): intra_cos={rq_results[f"L{l+1}"]["intra_cos"]:.3f}, '
              f'coverage={rq_results[f"L{l+1}"]["coverage"]:.3f}')

    # Oracle 1: Random balanced (均匀随机分配)
    print('\n[4] Oracle 1: 随机均衡 (Balanced Uniform)...')
    rng = np.random.default_rng(42)
    rand_results = {}
    for l in range(3):
        K = K_per_layer[l]
        # Balanced random: assign each item to random code, ensuring balance
        target_per_code = N // K
        codes = np.zeros(N, dtype=int)
        for c in range(K):
            start = c * target_per_code
            end = (c + 1) * target_per_code if c < K - 1 else N
            codes[start:end] = c
        rng.shuffle(codes)
        rand_results[f'L{l+1}'] = {
            'intra_cos': intra_code_cos(X, codes, K),
            'coverage': codebook_coverage(codes, K),
            'method': 'random_balanced',
        }
        print(f'  L{l+1} (K={K}): intra_cos={rand_results[f"L{l+1}"]["intra_cos"]:.3f}, '
              f'coverage={rand_results[f"L{l+1}"]["coverage"]:.3f}')

    # Oracle 2: Semantic K-means on full embedding
    print('\n[5] Oracle 2: 语义 K-means (full FLAN-T5 PCA-100)...')
    sem_results = {}
    for l in range(3):
        K = K_per_layer[l]
        km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048, n_init=3, max_iter=100)
        codes = km.fit_predict(X_pca)
        sem_results[f'L{l+1}'] = {
            'intra_cos': intra_code_cos(X, codes, K),
            'coverage': codebook_coverage(codes, K),
            'method': 'semantic_kmeans',
        }
        print(f'  L{l+1} (K={K}): intra_cos={sem_results[f"L{l+1}"]["intra_cos"]:.3f}, '
              f'coverage={sem_results[f"L{l+1}"]["coverage"]:.3f}')

    # Oracle 3: Semantic residual K-means (using r^l as input)
    print('\n[6] Oracle 3: 语义 Residual K-means (r^l as input)...')
    sem_res_results = {}
    for l in range(3):
        K = K_per_layer[l]
        r_l = r_lst[l].float().numpy()
        r_l_pca = PCA(n_components=min(50, r_l.shape[1]), random_state=42).fit_transform(r_l)
        km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048, n_init=3, max_iter=100)
        codes = km.fit_predict(r_l_pca)
        sem_res_results[f'L{l+1}'] = {
            'intra_cos': intra_code_cos(X, codes, K),
            'coverage': codebook_coverage(codes, K),
            'method': 'semantic_residual_kmeans',
        }
        print(f'  L{l+1} (K={K}): intra_cos={sem_res_results[f"L{l+1}"]["intra_cos"]:.3f}, '
              f'coverage={sem_res_results[f"L{l+1}"]["coverage"]:.3f}')

    # Oracle 4: Cumulative code (use q_total = sum codebooks[l] as input)
    print('\n[7] Oracle 4: 累计 codebook 输入 K-means...')
    q_cum = np.zeros_like(X)
    cum_results = {}
    for l in range(3):
        K = K_per_layer[l]
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum += q_l
        # K-means on q_cum
        q_cum_pca = PCA(n_components=50, random_state=42).fit_transform(q_cum)
        km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048, n_init=3, max_iter=100)
        codes = km.fit_predict(q_cum_pca)
        cum_results[f'L{l+1}'] = {
            'intra_cos': intra_code_cos(X, codes, K),
            'coverage': codebook_coverage(codes, K),
            'method': 'cumulative_codebook_kmeans',
        }
        print(f'  L{l+1} (K={K}): intra_cos={cum_results[f"L{l+1}"]["intra_cos"]:.3f}, '
              f'coverage={cum_results[f"L{l+1}"]["coverage"]:.3f}')

    # Summary
    print('\n[8] Summary...')
    methods = {
        'RQ_baseline': rq_results,
        'random_balanced': rand_results,
        'semantic_kmeans': sem_results,
        'semantic_residual_kmeans': sem_res_results,
        'cumulative_codebook_kmeans': cum_results,
    }
    out = {
        'n_items': N,
        'embed_path': EMBED_PT,
        'K_per_layer': K_per_layer,
        'note': 'Proxy: 用 intra-code cos sim 衡量每种 oracle 的语义一致性. 完整版需 TIGER 端到端 NDCG.',
        'methods': methods,
    }
    with open(os.path.join(OUT_DIR, 'oracle_methods.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/oracle_methods.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 306 Verdict: Oracle 深层 Code 上限 (proxy 版)\n\n')
        f.write(f'数据集: Toys (N={N}), 5 种 oracle × 3 层\n\n')
        f.write('## 5 Oracle 构造 (intra-code cos sim in FLAN-T5 space)\n\n')
        f.write('| Method | L1 | L2 | L3 |\n')
        f.write('|--------|-----|-----|-----|\n')
        for m_name, m_data in methods.items():
            row = []
            for l in range(3):
                v = m_data[f'L{l+1}']['intra_cos']
                row.append(f'{v:.3f}')
            f.write(f'| {m_name} | {" | ".join(row)} |\n')
        f.write('\n## 判读\n\n')
        # Compare RQ vs alternatives
        for l in range(3):
            l_name = f'L{l+1}'
            rq_v = rq_results[l_name]['intra_cos']
            sem_v = sem_results[l_name]['intra_cos']
            sem_res_v = sem_res_results[l_name]['intra_cos']
            cum_v = cum_results[l_name]['intra_cos']
            rand_v = rand_results[l_name]['intra_cos']
            best_alt = max(sem_v, sem_res_v, cum_v)
            best_alt_method = ['semantic_kmeans', 'semantic_residual_kmeans', 'cumulative_codebook_kmeans'][
                [sem_v, sem_res_v, cum_v].index(best_alt)
            ]
            f.write(f'### {l_name} (K={K_per_layer[l]})\n')
            f.write(f'- RQ baseline intra_cos = {rq_v:.3f}\n')
            f.write(f'- Best alternative = {best_alt:.3f} ({best_alt_method})\n')
            f.write(f'- Random baseline = {rand_v:.3f}\n')
            if best_alt > rq_v * 1.05:
                f.write(f'  → **Oracle 上限显著高于 RQ** → 深层位置有潜力, RQ 分配有改进空间\n')
            elif rq_v > best_alt * 1.02:
                f.write(f'  → **RQ 已接近 oracle 上限** → 深层位置信息饱和\n')
            else:
                f.write(f'  → RQ 与 oracle 相当 → 深层位置价值存疑, 需 NDCG 验证\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()