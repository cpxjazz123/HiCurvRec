"""
Task 64 — Exp1: 局部几何标签构造（向量化版）

高效计算三个局部几何度量：
  1. 有效维度（Participation Ratio）——各向异性
  2. 局部密度比（远/近邻距离比值）——曲率代理
  3. 角度散布（余弦相似度方差）——球面均匀度

无需逐四元组采样，全部基于预计算的邻域距离/余弦矩阵。

用法:
  PYTHONUNBUFFERED=1 python scripts/task6_exp1_geometry_prediction.py --device cuda:0
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_effective_dim(eigenvalues):
    """Participation ratio: (sum λ_i)² / sum(λ_i²). Lower = more anisotropic."""
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    if s2 < 1e-12:
        return 1.0
    return float(s * s / s2)


def compute_local_metrics_batch(neighbor_pts_gpu, distances_gpu):
    """
    Compute 3 geometric metrics for one neighborhood on GPU.

    Args:
        neighbor_pts_gpu: (K, D) tensor on GPU — the kNN neighbor embeddings
        distances_gpu: (K,) tensor on GPU — distances from query point to each neighbor

    Returns:
        (eff_dim, density_ratio, angular_spread) tuple
    """
    K, D = neighbor_pts_gpu.shape

    # 1. Effective dimensionality (anisotropy)
    centered = neighbor_pts_gpu - neighbor_pts_gpu.mean(dim=0, keepdim=True)
    # Fast SVD: use torch.svd_lowrank or just covariance eigendecomposition
    try:
        if K < D:
            # Compute on (K, K) Gram matrix for efficiency
            gram = centered @ centered.T  # (K, K)
            eigenvalues = torch.linalg.eigvalsh(gram).flip([0]).clamp(min=0)
        else:
            cov = (centered.T @ centered) / (K - 1)  # (D, D)
            eigenvalues = torch.linalg.eigvalsh(cov).flip([0]).clamp(min=0)
        eff_dim = compute_effective_dim(eigenvalues.cpu())
    except Exception:
        eff_dim = float(K)

    # 2. Local density ratio: mean(neighbor_dist) / min(neighbor_dist + 1e-8)
    # Higher = sparser neighborhood (points far from query but close to each other)
    # Lower = denser (all neighbors roughly same distance)
    d = distances_gpu  # (K,) — distances of kNN from the query point
    if K > 5:
        # Ratio of far (80th percentile) to near (20th percentile) distances
        sorted_d, _ = torch.sort(d)
        near = sorted_d[max(0, K // 5)].item()
        far = sorted_d[min(K - 1, 4 * K // 5)].item()
        density_ratio = far / max(near, 1e-8)
    else:
        density_ratio = 1.0

    # 3. Angular spread: std of pairwise cosine similarities
    # Normalize neighbor points
    norms = torch.norm(neighbor_pts_gpu, dim=1, keepdim=True).clamp(min=1e-8)
    unit = neighbor_pts_gpu / norms
    # Cosine similarity matrix (K, K) — use small batch to avoid OOM
    cos_sim = unit @ unit.T  # (K, K)
    # Exclude diagonal, compute std
    mask = ~torch.eye(K, dtype=torch.bool, device=neighbor_pts_gpu.device)
    off_diag = cos_sim[mask]
    angular_spread = float(off_diag.std().clamp(min=1e-8))

    return eff_dim, density_ratio, angular_spread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--k_neighbors", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_points", type=int, default=None, help="subsample for testing")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    print("=" * 60, flush=True)
    print("Task 64 — Exp1: 局部几何标签构造（向量化版）", flush=True)
    print("=" * 60, flush=True)

    # ── 加载数据 ──
    print(f"\n[1/4] Loading embeddings from {args.embeddings}...", flush=True)
    embeddings = torch.load(args.embeddings, map_location="cpu")
    N, D = embeddings.shape
    if args.max_points:
        embeddings = embeddings[:args.max_points]
        N = args.max_points
    print(f"  Shape: {list(embeddings.shape)}", flush=True)
    data_np = embeddings.numpy().astype(np.float32)

    # ── kNN 索引 ──
    k = min(args.k_neighbors, N)
    print(f"\n[2/4] Building kNN index (k={k})...", flush=True)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
    nn.fit(data_np)
    distances, indices = nn.kneighbors(data_np, n_neighbors=k)
    # distances: (N, k), indices: (N, k)
    print(f"  kNN built ({time.time()-t0:.1f}s)", flush=True)

    # ── 局部度量计算（GPU 批量化） ──
    print(f"\n[3/4] Computing local geometric signatures (N={N})...", flush=True)
    geo_labels = np.zeros((N, 3), dtype=np.float32)
    embeddings_gpu = embeddings.to(device)

    batch_size = 512  # points per GPU batch
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_indices = range(batch_start, batch_end)

        for local_i, global_i in enumerate(batch_indices):
            # Gather neighbor indices (exclude self: index 0 is the point itself)
            neigh_idx = torch.tensor(indices[global_i, 1:k], device=device)
            neigh_pts = embeddings_gpu[neigh_idx]  # (K-1, D)

            # Distances from query point to neighbors
            dists = torch.tensor(distances[global_i, 1:k], device=device)

            eff_dim, density_ratio, ang_spread = compute_local_metrics_batch(neigh_pts, dists)
            geo_labels[global_i] = [eff_dim, density_ratio, ang_spread]

        elapsed = time.time() - t0
        done = batch_end
        rate = done / max(elapsed, 0.1)
        eta = (N - done) / max(rate, 1)
        print(f"  {done}/{N} ({rate:.0f} pts/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.0f}s ({N/elapsed:.0f} pts/s)", flush=True)

    # ── 分析标签分布 ──
    print(f"\n[4/4] Analyzing geometric label distribution...", flush=True)

    # Normalize to z-scores then softmax
    geo_mean = geo_labels.mean(axis=0, keepdims=True)
    geo_std = geo_labels.std(axis=0, keepdims=True).clip(min=1e-8)
    geo_norm = (geo_labels - geo_mean) / geo_std
    # Softmax
    geo_exp = np.exp(geo_norm - geo_norm.max(axis=1, keepdims=True))
    geo_probs = geo_exp / geo_exp.sum(axis=1, keepdims=True)

    # Dominant type per point
    dominant = geo_probs.argmax(axis=1)

    print(f"\n  Geometric type distribution:", flush=True)
    for t_idx, name in enumerate(["High-Dim(0)", "Dense(1)", "Uniform(2)"]):
        count = int((dominant == t_idx).sum())
        print(f"    {name}: {count} ({100*count/N:.1f}%)", flush=True)

    # Mean raw metrics per type
    print(f"\n  Mean signature by predicted type:", flush=True)
    for t_idx, name in enumerate(["High-Dim", "Dense", "Uniform"]):
        mask = dominant == t_idx
        if mask.sum() == 0:
            continue
        mean_sig = geo_labels[mask].mean(axis=0)
        print(f"    {name}: [eff_dim={mean_sig[0]:.3f}, density_ratio={mean_sig[1]:.3f}, "
              f"ang_spread={mean_sig[2]:.5f}]", flush=True)

    # Label entropy
    entropies = -np.sum(geo_probs * np.log(geo_probs + 1e-12), axis=1)
    mean_entropy = float(entropies.mean())
    confident = float((entropies < 0.5).mean())
    print(f"\n  Mean label entropy: {mean_entropy:.4f}", flush=True)
    print(f"  Confident (entropy<0.5): {confident*100:.1f}%", flush=True)

    # Separability: intra-type vs inter-type
    type_means = {}
    for t in range(3):
        mask = dominant == t
        if mask.sum() > 0:
            type_means[t] = geo_labels[mask].mean(axis=0)

    inter_dists = []
    for t1 in type_means:
        for t2 in type_means:
            if t1 < t2:
                d = float(np.linalg.norm(type_means[t1] - type_means[t2]))
                inter_dists.append(d)
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0

    intra_std = float(geo_labels.std(axis=0).mean())  # average intra-type spread
    print(f"\n  Inter-type centroid dist: {mean_inter:.4f}", flush=True)
    print(f"  Intra-type avg std: {intra_std:.4f}", flush=True)

    separable = mean_inter > intra_std * 0.5
    print(f"  Separable? {'✅ YES' if separable else '❌ NO'}", flush=True)

    # ── 保存结果 ──
    out_dir = "products/task16/from_task64"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "config": {"N": N, "D": D, "k_neighbors": k, "batch_size": batch_size},
        "metric_names": ["effective_dim", "density_ratio", "angular_spread"],
        "type_distribution": {f"type_{t}": int((dominant == t).sum()) for t in range(3)},
        "metric_stats": {
            "eff_dim_mean": float(geo_labels[:, 0].mean()),
            "eff_dim_std": float(geo_labels[:, 0].std()),
            "density_ratio_mean": float(geo_labels[:, 1].mean()),
            "density_ratio_std": float(geo_labels[:, 1].std()),
            "ang_spread_mean": float(geo_labels[:, 2].mean()),
            "ang_spread_std": float(geo_labels[:, 2].std()),
        },
        "mean_label_entropy": mean_entropy,
        "confident_fraction": confident,
        "inter_type_centroid_dist": mean_inter,
        "intra_type_avg_std": intra_std,
        "separability_verdict": separable,
        "prediction_is_feasible": separable,
    }

    out_path = os.path.join(out_dir, "exp1_geo_labels.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)

    # Save geo labels for Exp2
    labels_path = os.path.join(out_dir, "exp1_geo_labels.npy")
    np.save(labels_path, geo_probs)
    print(f"Geo labels saved: {labels_path}", flush=True)

    # ── Final verdict ──
    print("\n" + "=" * 60, flush=True)
    print("Exp1 Verdict", flush=True)
    print("=" * 60, flush=True)
    if separable:
        print("✅ 局部几何特性可区分 — 编码器有可能学会预测几何标签", flush=True)
    else:
        print("❌ 局部几何特性混杂不可区分 — 几何标签路线否证", flush=True)
    print(f"Acc_geo_proxy (entropy<0.5): {confident*100:.1f}% (目标 > 75%)", flush=True)


if __name__ == "__main__":
    main()
