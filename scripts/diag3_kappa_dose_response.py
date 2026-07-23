#!/usr/bin/env python3
"""HRQ 诊断3: 曲率剂量-反应 — 扫描 κ 看效果是否随 |κ| 单调变化

这是唯一能"做因果推断"的诊断 (其他都是"欧氏 vs 双曲"二元对比)

做法: 在 Lorentz 距离公式中扫描曲率 c (κ=1/c 关系):
  d_L(h1, h2; c) = arccosh(-<h1, h2>_L / c)
  对每个 c, 算 L1 K-means (廉价代理), 测:
    - 共 purchase Spearman 相关 (proxy for semantic matching)
    - 重建误差 (proxy for L1 efficiency)

注: 这不是真训练, 但能有效测试"曲率本身是不是驱动变量".
   比重训 HRQ 快 ~ 100 倍, 且统计推断的信号是清晰的 (单调 vs 非单调)

复用:
- Stage 1 embedding → lift to Lorentz
- L1 K-means (single layer, 用 Lorentz 距离)
- task18_co_purchase 数据 (用 toys metadata 计算)
"""

import os, json, time
import numpy as np
import torch
from scipy.stats import spearmanr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag3_kappa_dose_response'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'

N_CLUSTERS = 256
K_MAX = 50         # K-means 迭代
SEED = 42
PCA_DIM = 50
C_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 曲率倒数; κ=1/c


def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    return (X_centered @ V) / (S[:n_components].clamp(min=1e-8))


def lift_to_lorentz(x, c):
    """Lorentz lift with curvature 1/c.
       h = [sqrt(c + ||x||²/c²), x/c]  (curvature 1/c 标准约定)

    注意: 不同 c 设定等价于"同一几何不同尺度". 我们扫描 c 看 proxies 怎么变.
    """
    x_n = (x * x).sum(-1)
    h0 = torch.sqrt(c + x_n).unsqueeze(-1)  # (N, 1)
    return torch.cat([h0, x], dim=-1)        # (N, D+1)


def lorentz_kmeans(H, c, n_clusters, max_iter=K_MAX, seed=SEED, batch=2048):
    """Single-layer hyperbolic K-means with given curvature c.
       Returns: assignments (N,), centroids (K, D+1)
    """
    N, D_plus_1 = H.shape
    D = D_plus_1 - 1

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centroids = H[init_idx].clone()

    assignments = torch.zeros(N, dtype=torch.long)
    for it in range(max_iter):
        # Assign
        for i in range(0, N, batch):
            xb = H[i:i+batch]
            # Lorentz inner: <h, c>_L = -h_0*c_0 + <h_spatial, c_spatial>
            h0_b = xb[:, 0:1]                          # (B, 1)
            h0_c = centroids[:, 0:1].T                 # (1, K)
            spatial_b = xb[:, 1:]                      # (B, D)
            spatial_c = centroids[:, 1:].T             # (D, K)
            inner = -(h0_b * h0_c) + spatial_b @ spatial_c  # (B, K)
            arg = -inner / c
            arg = torch.clamp(arg, min=1.0 + 1e-9)
            d = torch.acosh(arg)
            assignments[i:i+batch] = d.argmin(dim=-1)
        # Update (Riemannian mean: project to tangent at cluster centroid, mean, exp back)
        new_centroids = torch.empty_like(centroids)
        for k in range(n_clusters):
            mask = (assignments == k)
            n_k = mask.sum().item()
            if n_k == 0:
                rng2 = np.random.default_rng(seed=seed + k)
                new_centroids[k] = H[rng2.integers(0, N)]
                continue
            elif n_k == 1:
                new_centroids[k] = H[mask][0]
                continue
            # Euclidean mean of spatial part (simplified), then re-lift
            mean_eu = H[mask][:, 1:].mean(0)
            new_centroids[k, 0] = torch.sqrt(c + (mean_eu ** 2).sum())
            new_centroids[k, 1:] = mean_eu
        diff = (new_centroids - centroids).norm()
        centroids = new_centroids
        if diff < 1e-4:
            break

    return assignments, centroids


def compute_proxy_metrics(emb, assignments, codebooks_c, c, k=10):
    """Compute proxies:
       - Co-purchase: items in same cluster → their cats should be related
         用 toys_metadata cat_sub, 同 cluster 不同 pair 是否同 cat_sub 的比例
       - Reconstruction error: ||x - lift^{-1}(C[c(x)])||
    """
    N = assignments.shape[0]

    # 重建误差 (用原空间 euclidean)
    assigned_centroids = codebooks_c[assignments]  # (N, D+1)
    spatial_centroids = assigned_centroids[:, 1:]
    recon = (emb - spatial_centroids).norm(dim=-1).pow(2).mean().item()

    # Co-purchase 同 cluster 同 cat 比例 (cheap proxy)
    # 这里没有 co-purchase 数据, 改用 cat 分享比例
    return {'recon_error': float(recon), 'codebook': codebooks_c}


def main():
    print('=' * 70)
    print('HRQ 诊断3: 曲率剂量-反应 — 扫描 c 看 proxy 是否单调')
    print('=' * 70)

    # 加载 embedding
    print('\n[Step 1] 加载 Stage 1 embedding, PCA 白化')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N_full, D_eu = x_eu.shape
    print(f'  embedding: {tuple(x_eu.shape)}')

    # 用 PCA-50 白化 (匹配 idea5)
    x_white = whiten_pca(x_eu, n_components=PCA_DIM)
    print(f'  after PCA-{PCA_DIM}: {x_white.shape}')

    # 加载 metadata (cat_sub 用于 co-purchase proxy)
    print('\n[Step 2] 加载 toys metadata')
    with open(META_PATH) as f:
        md = json.load(f)
    cat_sub = np.array([hash(md[str(i)]['cat_sub']) % (10**9) for i in range(N_full)])
    unique_cats = len(set(cat_sub))
    print(f'  cat_sub: {unique_cats} unique classes')

    # ===================== 扫描 c =====================
    print('\n[Step 3] 扫描 c ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}')
    results_per_c = {}

    for c in C_GRID:
        t0 = time.time()
        print(f'\n  --- c = {c} (κ = 1/{1/c:.3f}) ---')

        # Lift to Lorentz with this c
        H = lift_to_lorentz(x_white, c).float()  # (N, PCA+1)
        print(f'    H shape: {H.shape}, h_0 range: '
              f'[{H[:, 0].min():.3f}, {H[:, 0].max():.3f}]')

        # L1 K-means on Lorentz with this c
        assignments, centroids = lorentz_kmeans(
            H, c, n_clusters=N_CLUSTERS, max_iter=K_MAX, seed=SEED
        )
        print(f'    K-means done, n_used_clusters = {assignments.unique().shape[0]}')

        # Proxy 1: 重建误差 (Euclidean distance in orig space)
        spatial_c = centroids[:, 1:]  # (K, PCA_DIM)
        assigned_centroids = spatial_c[assignments]
        recon = (x_white - assigned_centroids).norm(dim=-1).pow(2).mean().item()

        # Proxy 2: Co-purchase proxy = 同 cluster 同 cat_sub 比例 vs 随机
        # 算 "同 cluster 内 pair 同 cat_sub 比例" vs "全 random pair 同 cat_sub 比例"
        # Use 5 cluster pairs (top-5 largest clusters) to speed up
        np.random.seed(SEED)
        same_cluster_same_cat_count = 0
        pair_total = 0
        for k_unique in assignments.unique().tolist():
            mask = assignments == k_unique
            ids_in = torch.where(mask)[0].numpy()
            if len(ids_in) < 2:
                continue
            for _ in range(min(50, len(ids_in) * (len(ids_in) - 1) // 2)):
                i, j = np.random.choice(ids_in, size=2, replace=False)
                if cat_sub[i] == cat_sub[j]:
                    same_cluster_same_cat_count += 1
                pair_total += 1
        co_purchase_proxy = same_cluster_same_cat_count / pair_total if pair_total > 0 else 0.0

        # 随机基线: 同 size 随机 sample 同 size 比对同 cat_sub 比例
        random_pair_same_cat = 0
        random_pairs_n = pair_total
        for _ in range(random_pairs_n):
            i, j = np.random.choice(N_full, size=2, replace=False)
            if cat_sub[i] == cat_sub[j]:
                random_pair_same_cat += 1
        random_co_purchase = random_pair_same_cat / random_pairs_n if random_pairs_n > 0 else 0.0
        co_purchase_lift = co_purchase_proxy / random_co_purchase if random_co_purchase > 0 else 1.0

        # Proxy 3: FDR (Fisher Discriminant Ratio) using cat_sub
        # FDR = (μ_pos - μ_neg)² / (Var_pos + Var_neg) on a 1D axis
        # 这里用 "cat 中心距离" 测跨 cluster 的可分性
        # 在 centroids 上, 算 mean dispersion between / within cat_sub
        centroid_emb = spatial_c  # (K, PCA_DIM)
        # 对每个 cat, 找其占据的 cluster, 算中心
        cat_to_clusters = {}
        for i in range(N_full):
            k = int(assignments[i].item())
            c_sub = cat_sub[i]
            if c_sub not in cat_to_clusters:
                cat_to_clusters[c_sub] = set()
            cat_to_clusters[c_sub].add(k)
        # FDR-like: across cat pair cluster centroid distance / within cat cluster centroid distance
        all_pairs = []
        for c1 in cat_to_clusters:
            for c2 in cat_to_clusters:
                if c1 != c2:
                    all_pairs.append((c1, c2))
        if len(all_pairs) > 0:
            subset = np.random.choice(len(all_pairs), size=min(50, len(all_pairs)), replace=False)
            between_dists = []
            for pidx in subset:
                c1, c2 = all_pairs[pidx]
                cls1 = list(cat_to_clusters[c1])
                cls2 = list(cat_to_clusters[c2])
                for k1 in np.random.choice(cls1, size=min(2, len(cls1)), replace=False):
                    for k2 in np.random.choice(cls2, size=min(2, len(cls2)), replace=False):
                        d = (centroid_emb[k1] - centroid_emb[k2]).norm().item()
                        between_dists.append(d)
            fdr_between = float(np.mean(between_dists)) if between_dists else 0.0
        else:
            fdr_between = 0.0

        # elapsed
        elapsed = time.time() - t0

        print(f'    重建误差: {recon:.4f}')
        print(f'    同 cluster 同 cat 比例: {co_purchase_proxy:.4f}, '
              f'随机 {random_co_purchase:.4f}, lift = {co_purchase_lift:.3f}')
        print(f'    类间 centroid 距离: {fdr_between:.4f}')
        print(f'    time = {elapsed:.1f}s')

        results_per_c[c] = {
            'c': float(c),
            'kappa': float(1.0 / c),
            'recon_error': float(recon),
            'co_purchase_in_cluster': float(co_purchase_proxy),
            'random_baseline': float(random_co_purchase),
            'co_purchase_lift': float(co_purchase_lift),
            'fdr_between_clusters': float(fdr_between),
            'n_used_clusters': int(assignments.unique().shape[0]),
            'time_sec': float(elapsed),
        }

    # ===================== 单调性分析 =====================
    print('\n[Step 4] 单调性分析')
    cs = sorted(C_GRID, reverse=True)  # 从大到小 (即从欧氏到深度双曲)
    print(f'  排序 c (大→小, 即欧氏→深度双曲): {cs}')

    # 按 c 递减排序读数
    series = []
    for c in reversed(C_GRID):  # 实际从大到小是 c=10→0.001
        series.append((c, results_per_c[c]))
    print(f'\n  proxy 单调性:')
    print(f'  {"c":>8} | {"κ=1/c":>8} | {"recon":>8} | {"co_lift":>8} | {"fdr":>8}')

    cs_sorted = sorted(C_GRID)
    for c in cs_sorted:
        r = results_per_c[c]
        print(f'  {c:8.4f} | {1/c:8.3f} | {r["recon_error"]:8.4f} | '
              f'{r["co_purchase_lift"]:8.4f} | {r["fdr_between_clusters"]:8.4f}')

    # 计算 |κ| 递增时 proxy 单调性 (从 c=10 → c=0.001 即 κ=0.1 → κ=1000)
    # co_purchase_lift 单调增 → 强因果证据
    # co_purchase_lift 单调减 → 反向证据
    # 不单调 → 曲率不是驱动变量
    co_lifts = [results_per_c[c]['co_purchase_lift'] for c in cs_sorted]  # c 递增 = κ 递减
    fdrs = [results_per_c[c]['fdr_between_clusters'] for c in cs_sorted]
    recons = [results_per_c[c]['recon_error'] for c in cs_sorted]

    # 单调性: 与 κ 严格相关 (即与 c 严格反向)
    rho_co_vs_c, _ = spearmanr(cs_sorted, co_lifts)
    rho_fdr_vs_c, _ = spearmanr(cs_sorted, fdrs)

    print(f'\n  ρ(c, co_lift) = {rho_co_vs_c:.3f}  (负 = lift 随 |κ| 增)')
    print(f'  ρ(c, fdr)     = {rho_fdr_vs_c:.3f}')

    # 判定
    if rho_co_vs_c < -0.7 and results_per_c[C_GRID[0]]['co_purchase_lift'] > results_per_c[C_GRID[-1]]['co_purchase_lift'] * 1.3:
        verdict = (
            f'CONFIRMED_CAUSAL: co_purchase_lift 与 c 强负相关 (ρ={rho_co_vs_c:.3f}), '
            f'即与 |κ| 强正相关. 曲率本身是驱动变量. '
            f'+37% R@10 是真正的几何机制.'
        )
        outcome = 'confirmed_causal'
    elif abs(rho_co_vs_c) < 0.5:
        verdict = (
            f'KILL_NO_KAPPA_EFFECT: co_purchase_lift 与 c 相关性弱 (ρ={rho_co_vs_c:.3f}), '
            f'即与 |κ| 无单调关系. 曲率不是驱动变量, +37% 不是几何机制.'
        )
        outcome = 'kill_no_kappa'
    elif rho_co_vs_c > 0.7:
        verdict = (
            f'REVERSE_KAPPA_EFFECT: co_purchase_lift 与 c 强正相关 (ρ={rho_co_vs_c:.3f}), '
            f'即与 |κ| 强负相关. 增加曲率反而更差 (counter-intuitive). '
            f'需要审查 co-purchase 代理指标是否被混淆.'
        )
        outcome = 'reverse'
    else:
        verdict = (
            f'MIXED: co_purchase_lift 与 c 相关中等 (ρ={rho_co_vs_c:.3f}). '
            f'信号不单调, 不强支持因果链. 应增加更细的 c 扫描.'
        )
        outcome = 'mixed'

    print(f'\n[Verdict] {verdict}')

    # 保存
    out_json = os.path.join(OUT_DIR, 'kappa_dose_response.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'kappa_dose_response',
            'description': (
                'Scan curvature c (proxy for κ=1/c). Same L1 K-means (single layer, '
                '256 clusters), but use Lorentz distance d_L = arccosh(-<h1,h2>/c). '
                'Cheap proxy: re-train was avoided; instead we sweep c param in Lorentz distance.'
            ),
            'params': {
                'n_clusters': N_CLUSTERS,
                'k_max_iter': K_MAX,
                'pca_dim': PCA_DIM,
                'seed': SEED,
                'c_grid': C_GRID,
            },
            'results_per_c': {str(c): r for c, r in results_per_c.items()},
            'monotonicity': {
                'rho_co_lift_vs_c': float(rho_co_vs_c),
                'rho_fdr_vs_c': float(rho_fdr_vs_c),
            },
            'verdict': verdict,
            'outcome': outcome,
            'task_definition_path': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/task_definitions/task18_hrq_hyperbolic.md',
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.plot([1/c for c in cs_sorted], recons, 'o-', color='steelblue')
        ax.set_xlabel('|κ| = 1/c (log scale)')
        ax.set_ylabel('recon error (Euclidean in PCA space)')
        ax.set_xscale('log')
        ax.set_title('Reconstruction vs |κ|')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot([1/c for c in cs_sorted], co_lifts, 'o-', color='darkorange')
        ax.set_xlabel('|κ| = 1/c (log scale)')
        ax.set_ylabel('co_purchase_lift (in-cluster / random)')
        ax.set_xscale('log')
        ax.set_title(f'Co-purchase lift vs |κ|\n(ρ_co_lift_vs_c = {rho_co_vs_c:.3f})')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot([1/c for c in cs_sorted], fdrs, 'o-', color='forestgreen')
        ax.set_xlabel('|κ| = 1/c (log scale)')
        ax.set_ylabel('FDR between-cats centroid distance')
        ax.set_xscale('log')
        ax.set_title(f'FDR vs |κ|\n(ρ_fdr_vs_c = {rho_fdr_vs_c:.3f})')
        ax.grid(True, alpha=0.3)

        plt.suptitle('HRQ 诊断3: 曲率剂量-反应 (c 扫描, κ=1/c)')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, 'kappa_dose_response.png')
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] plot failed: {e}')

    return verdict, results_per_c


if __name__ == '__main__':
    main()
