"""
Task 62 — Exp2: 残差几何特性度量
  - Gromov δ-Hyperbolicity (双曲性)
  - Angular Uniformity (球面均匀性)
  - Isotropy Score (各向同性)

用法: python scripts/task6_measure_geometry.py [--rqvae_ckpt products/task16/from_task62/rqvae_ckpt.pt]
"""

import os, sys, argparse, json, math
import torch
import numpy as np
from scipy.stats import beta, entropy, kstest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def compute_delta_hyperbolicity(points, n_samples=1000, sample_size=1000, seed=42):
    """
    计算 Gromov δ-Hyperbolicity。
    points: (N, D) tensor
    返回: {delta_worst, delta_avg, delta_rel}
    """
    N = points.shape[0]
    if N > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, sample_size, replace=False)
        points = points[idx]

    n = points.shape[0]
    rng = np.random.RandomState(seed + 1)
    deltas = []

    for _ in range(n_samples):
        # 随机采样 4 个点
        idx = rng.choice(n, 4, replace=False)
        x, y, z, w = points[idx]

        # 欧氏距离
        d_xy = torch.norm(x - y).item()
        d_xz = torch.norm(x - z).item()
        d_xw = torch.norm(x - w).item()
        d_yz = torch.norm(y - z).item()
        d_yw = torch.norm(y - w).item()
        d_zw = torch.norm(z - w).item()

        # 4 点条件中的三个和
        d1 = d_xy + d_zw
        d2 = d_xz + d_yw
        d3 = d_xw + d_yz

        # 排序: a ≤ b ≤ c
        a, b, c = sorted([d1, d2, d3])
        delta = (c - b) / 2.0
        deltas.append(delta)

    deltas = np.array(deltas)

    # 归一化: diameter
    # 用采样的最大 pairwise distance 近似
    max_dist = 0.0
    for _ in range(min(500, n)):
        i, j = rng.choice(n, 2, replace=False)
        d = torch.norm(points[i] - points[j]).item()
        if d > max_dist:
            max_dist = d
    max_dist = max(max_dist, 1e-8)

    return {
        "delta_worst": float(np.max(deltas)),
        "delta_avg": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_rel": float(np.mean(deltas) / max_dist),
        "delta_rel_worst": float(np.max(deltas) / max_dist),
        "diameter": float(max_dist),
    }


@torch.no_grad()
def compute_angular_uniformity(points, sample_size=2000, seed=42):
    """
    计算 Angular Uniformity: 余弦相似度分布 vs 理论球面均匀分布。
    points: (N, D) tensor
    返回: {uniformity_score, kl_divergence, ks_stat, ks_pvalue}
    """
    N, D = points.shape

    # 归一化到单位球
    norms = torch.norm(points, dim=1, keepdim=True)
    norms = norms.clamp(min=1e-8)
    unit = points / norms

    # 采样点对计算余弦相似度
    rng = np.random.RandomState(seed)
    n_pairs = min(N * (N - 1) // 2, 50000)
    idx1 = rng.choice(N, n_pairs, replace=True)
    idx2 = rng.choice(N, n_pairs, replace=True)

    cos_sim = (unit[idx1] * unit[idx2]).sum(dim=1).cpu().numpy()
    cos_sim = np.clip(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)

    # 理论分布: 在 D 维球面上均匀分布时, cos sim 的分布
    # Beta((D-1)/2, (D-1)/2) on (-1, 1) 映射到 [0,1]
    # 实际 Beta 在 [0,1]；我们映射 cos sim 从 [-1,1] 到 [0,1]
    a_param = (D - 1) / 2.0
    b_param = (D - 1) / 2.0

    cos_norm = (cos_sim + 1) / 2.0  # -> [0, 1]
    cos_norm = np.clip(cos_norm, 1e-8, 1 - 1e-8)

    # KL 散度: 用直方图近似
    bins = 100
    hist_emp, edges = np.histogram(cos_norm, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    pdf_theoretical = beta.pdf(bin_centers, a_param, b_param)
    pdf_theoretical /= pdf_theoretical.sum() * (bin_centers[1] - bin_centers[0])

    # KL(P || Q): P = empirical, Q = theoretical
    eps = 1e-12
    kl = entropy(hist_emp + eps, pdf_theoretical + eps)

    # KS test
    ks_stat, ks_pval = kstest(cos_norm, 'beta', args=(a_param, b_param))

    return {
        "uniformity_score": float(-kl),  # 越大越好（越接近球面均匀）
        "kl_divergence": float(kl),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "dimension": D,
    }


@torch.no_grad()
def compute_isotropy_score(points):
    """
    计算各向同性 Score: λ_min / λ_max
    points: (N, D) tensor
    返回: {isotropy_score, condition_number, explained_var_ratio_1st}
    """
    # 中心化
    centered = points - points.mean(dim=0, keepdim=True)
    N = centered.shape[0]

    # 协方差矩阵 (D, D)
    cov = (centered.T @ centered) / (N - 1)

    # 特征值
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-12)

    lambda_min = eigenvalues[0].item()
    lambda_max = eigenvalues[-1].item()

    isotropy = lambda_min / lambda_max if lambda_max > 0 else 0.0
    cond_num = lambda_max / lambda_min if lambda_min > 0 else float('inf')

    # 首成分解释方差比
    total_var = eigenvalues.sum().item()
    var_ratio_1st = eigenvalues[-1].item() / total_var if total_var > 0 else 0.0

    return {
        "isotropy_score": float(isotropy),
        "condition_number": float(cond_num),
        "explained_var_ratio_1st": float(var_ratio_1st),
        "eigenvalue_min": float(lambda_min),
        "eigenvalue_max": float(lambda_max),
    }


def extract_residuals(model, embeddings, device="cpu"):
    """提取 RQ-VAE 三层残差"""
    model.eval()
    model = model.to(device)
    embeddings = embeddings.to(device)

    residual = embeddings
    residuals_list = []

    with torch.no_grad():
        for q in model.quantizers:
            dist = torch.cdist(residual, q.codebook, p=2)
            ind = dist.argmin(dim=1)
            z_q = q.codebook[ind]
            r = residual - z_q
            residuals_list.append(r.cpu())
            residual = residual - z_q

    return residuals_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae_ckpt", default="products/task16/from_task62/rqvae_ckpt.pt")
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=1000, help="δ 采样子集大小")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 62 — Exp2: 残差几何特性度量")
    print("=" * 60)

    if not os.path.exists(args.rqvae_ckpt):
        print(f"ERROR: {args.rqvae_ckpt} not found.")
        sys.exit(1)

    # Load model
    print(f"\n[1/3] Loading model from {args.rqvae_ckpt}...")
    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu")
    from scripts.task62_train_rqvae import ResidualQuantization
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])

    # Load embeddings
    print(f"[2/3] Loading embeddings...")
    embeddings = torch.load(args.embeddings, map_location="cpu")

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[3/3] Extracting residuals (device={device})...")
    residuals = extract_residuals(model, embeddings, device=device)

    # Include the original embedding as "raw" for comparison
    all_sets = {"Raw (original)": embeddings.cpu()}
    for i, r in enumerate(residuals):
        all_sets[f"L{i+1}残差"] = r

    # Run geometric analyses
    results = {}
    for name, pts in all_sets.items():
        print(f"\n--- {name} ---")
        geo = {}

        print("  δ-Hyperbolicity...")
        geo["delta"] = compute_delta_hyperbolicity(pts, n_samples=args.n_samples)

        print("  Angular Uniformity...")
        geo["angular_uniformity"] = compute_angular_uniformity(pts)

        print("  Isotropy Score...")
        geo["isotropy"] = compute_isotropy_score(pts)

        results[name] = geo

        print(f"    δ_rel = {geo['delta']['delta_rel']:.4f}")
        print(f"    Angular Uniformity = {geo['angular_uniformity']['uniformity_score']:.4f}")
        print(f"    Isotropy = {geo['isotropy']['isotropy_score']:.4f}")

    # Save
    out_dir = "products/task16/from_task62"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
