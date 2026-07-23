"""
Task 63 — Exp1: 码书在不同流形上学习后的结构差异

在相同的残差数据上，用三种不同优化方式（欧氏SGD / 双曲黎曼SGD / 球面黎曼SGD）
训练码书，然后度量其几何特性与差异度。

用法: python scripts/task6_exp1_codebook_diff.py [--device cuda:0]
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from scipy.stats import beta, entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── 几何度量工具（复用 task62 的逻辑） ─────────────────────────────


def compute_isotropy(points):
    """Isotropy score = λ_min / λ_max"""
    centered = points - points.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    return (eigenvalues[0] / eigenvalues[-1]).item()


def compute_gromov_delta(points, n_samples=1000, seed=42):
    """Gromov δ-Hyperbolicity (sampled)."""
    n = points.shape[0]
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_samples):
        idx = rng.choice(n, 4, replace=False)
        x, y, z, w = points[idx]
        d_xy = torch.norm(x - y).item()
        d_xz = torch.norm(x - z).item()
        d_xw = torch.norm(x - w).item()
        d_yz = torch.norm(y - z).item()
        d_yw = torch.norm(y - w).item()
        d_zw = torch.norm(z - w).item()
        d1, d2, d3 = sorted([d_xy + d_zw, d_xz + d_yw, d_xw + d_yz])
        deltas.append((d3 - d2) / 2.0)
    # Diameter
    max_dist = 0.0
    for _ in range(min(500, n)):
        i, j = rng.choice(n, 2, replace=False)
        d = torch.norm(points[i] - points[j]).item()
        max_dist = max(max_dist, d)
    max_dist = max(max_dist, 1e-8)
    delta_avg = np.mean(deltas)
    return {"delta_avg": float(delta_avg), "delta_rel": float(delta_avg / max_dist), "diameter": float(max_dist)}


def compute_angular_uniformity(points, seed=42):
    """Angular Uniformity: KL(empirical_cos || Beta((D-1)/2, (D-1)/2))."""
    N, D = points.shape
    norms = torch.norm(points, dim=1, keepdim=True).clamp(min=1e-8)
    unit = points / norms
    rng = np.random.RandomState(seed)
    n_pairs = min(N * (N - 1) // 2, 50000)
    idx1 = rng.choice(N, n_pairs, replace=True)
    idx2 = rng.choice(N, n_pairs, replace=True)
    cos_sim = (unit[idx1] * unit[idx2]).sum(dim=1).cpu().numpy()
    cos_sim = np.clip(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)
    a_param = (D - 1) / 2.0
    cos_norm = np.clip((cos_sim + 1) / 2.0, 1e-8, 1 - 1e-8)
    bins = 100
    hist_emp, edges = np.histogram(cos_norm, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    pdf_theory = beta.pdf(bin_centers, a_param, a_param)
    pdf_theory /= pdf_theory.sum() * (bin_centers[1] - bin_centers[0])
    kl = entropy(hist_emp + 1e-12, pdf_theory + 1e-12)
    return {"uniformity_score": float(-kl), "kl_divergence": float(kl)}


# ─── 码书训练 ────────────────────────────────────────────


def train_euclidean(data, K=256, epochs=10, lr=0.01, device="cpu"):
    """Standard SGD codebook training (Euclidean)."""
    N, D = data.shape
    # K-means initialization
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024)
    kmeans.fit(data.cpu().numpy())
    codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device, requires_grad=True)
    data_d = data.to(device)

    opt = torch.optim.SGD([codebook], lr=lr)
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        shuffled = data_d[perm]
        total_loss = 0.0
        for i in range(0, N, 1024):
            batch = shuffled[i:i + 1024]
            dist = torch.cdist(batch, codebook, p=2)
            indices = dist.argmin(dim=1)
            z_q = codebook[indices]
            loss = (batch - z_q).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / N
        history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    return codebook.detach(), history


def train_poincare(data, K=256, epochs=10, lr=0.01, device="cpu"):
    """RiemannianSGD on Poincaré ball."""
    import geoopt
    N, D = data.shape
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024)
    kmeans.fit(data.cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    # Project to Poincaré ball
    ball = geoopt.PoincareBall(c=1.0)
    centroids = ball.projx(centroids * 0.1)  # scale down for safe projection

    manifold_param = geoopt.ManifoldParameter(centroids, manifold=ball)
    opt = geoopt.optim.RiemannianSGD([manifold_param], lr=lr)
    data_d = data.to(device)
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        shuffled = data_d[perm]
        total_loss = 0.0
        for i in range(0, N, 1024):
            batch = shuffled[i:i + 1024]
            # Use Euclidean distance for assignment (same across all settings)
            dist = torch.cdist(batch, manifold_param, p=2)
            indices = dist.argmin(dim=1)
            z_q = manifold_param[indices]
            loss = (batch - z_q).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / N
        history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    return manifold_param.detach(), history


def train_sphere(data, K=256, epochs=10, lr=0.01, device="cpu"):
    """RiemannianSGD on Sphere."""
    import geoopt
    N, D = data.shape
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024)
    kmeans.fit(data.cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    sphere = geoopt.Sphere()
    centroids = sphere.projx(centroids)

    manifold_param = geoopt.ManifoldParameter(centroids, manifold=sphere)
    opt = geoopt.optim.RiemannianSGD([manifold_param], lr=lr)
    data_d = data.to(device)
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        shuffled = data_d[perm]
        total_loss = 0.0
        for i in range(0, N, 1024):
            batch = shuffled[i:i + 1024]
            dist = torch.cdist(batch, manifold_param, p=2)
            indices = dist.argmin(dim=1)
            z_q = manifold_param[indices]
            loss = (batch - z_q).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / N
        history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    return manifold_param.detach(), history


# ─── 码书差异度 ──────────────────────────────────────────


def compute_codebook_difference(C_list, names):
    """
    Compute pairwise Frobenius norm of cosine similarity matrices.
    C_list: list of (K, D) tensors
    returns: {Δ(A,B): value} dict and similarity matrices
    """
    sim_mats = {}
    for i, (C, name) in enumerate(zip(C_list, names)):
        # Normalize each codeword
        norms = torch.norm(C, dim=1, keepdim=True).clamp(min=1e-8)
        C_norm = C / norms
        S = (C_norm @ C_norm.T).cpu().numpy()  # (K, K)
        sim_mats[name] = S

    diffs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = np.linalg.norm(sim_mats[names[i]] - sim_mats[names[j]], 'fro')
            diffs[f"Δ({names[i]},{names[j]})"] = float(diff)

    return diffs, sim_mats


# ─── 主流程 ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("Task 63 — Exp1: 码书结构差异分析")
    print("=" * 60)

    # ── 加载数据 ──
    print(f"\n[1/4] Loading embeddings from {args.embeddings}...")
    embeddings = torch.load(args.embeddings, map_location="cpu")
    print(f"  Shape: {list(embeddings.shape)}")
    print(f"  Norms: min={embeddings.norm(dim=1).min():.4f}, mean={embeddings.norm(dim=1).mean():.4f}")

    # Use L2-normalized data (as-is, already normalized)
    data = embeddings

    # ── 训练三个码书 ──
    print(f"\n[2/4] Training 3 codebooks (K={args.K}, epochs={args.epochs}, lr={args.lr})...")

    trainers = [
        ("Euclidean", train_euclidean),
        ("Poincare", train_poincare),
        ("Sphere", train_sphere),
    ]

    codebooks = {}
    histories = {}
    for name, trainer_fn in trainers:
        print(f"\n  Training {name}...")
        t0 = time.time()
        cb, hist = trainer_fn(data, K=args.K, epochs=args.epochs, lr=args.lr, device=device)
        elapsed = time.time() - t0
        print(f"  Done ({elapsed:.1f}s)")
        codebooks[name] = cb.cpu()
        histories[name] = hist

    # ── 几何度量 ──
    print(f"\n[3/4] Computing geometric metrics...")
    metrics = {}
    for name, cb in codebooks.items():
        print(f"\n  {name} codebook metrics:")
        m = {}

        m["isotropy"] = compute_isotropy(cb)
        print(f"    Isotropy = {m['isotropy']:.6f}")

        delta_info = compute_gromov_delta(cb, n_samples=500)
        m["delta_rel"] = delta_info["delta_rel"]
        m["delta_avg"] = delta_info["delta_avg"]
        m["diameter"] = delta_info["diameter"]
        print(f"    δ_rel = {m['delta_rel']:.6f}")

        uniformity = compute_angular_uniformity(cb)
        m["uniformity_score"] = uniformity["uniformity_score"]
        m["kl_divergence"] = uniformity["kl_divergence"]
        print(f"    Angular Uniformity = {m['uniformity_score']:.4f}")

        metrics[name] = m

    # ── 码书差异度 ──
    print(f"\n[4/4] Computing codebook differences...")
    names = list(codebooks.keys())
    cb_list = [codebooks[n] for n in names]
    diffs, sim_mats = compute_codebook_difference(cb_list, names)
    for pair, val in diffs.items():
        print(f"  {pair}: {val:.4f}")

    # ── 保存结果 ──
    out_dir = "products/task16/from_task63"
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    results = {
        "config": {"K": args.K, "epochs": args.epochs, "lr": args.lr},
        "training_history": histories,
        "metrics": metrics,
        "codebook_differences": diffs,
    }
    out_path = os.path.join(out_dir, "exp1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Save codebooks
    ckpt_path = os.path.join(out_dir, "exp1_codebooks.pt")
    torch.save(codebooks, ckpt_path)
    print(f"Codebooks saved: {ckpt_path}")

    # ── 判定 ──
    print("\n" + "=" * 60)
    print("Exp1 Summary")
    print("=" * 60)
    print(f"{'Metric':>20}", end="")
    for n in names:
        print(f"  {n:>12}", end="")
    print()
    for metric_key in ["isotropy", "delta_rel", "uniformity_score"]:
        print(f"{metric_key:>20}", end="")
        for n in names:
            val = metrics[n].get(metric_key, 0)
            print(f"  {val:>12.6f}", end="")
        print()
    print()
    for pair, val in diffs.items():
        print(f"  {pair} = {val:.4f}")


if __name__ == "__main__":
    main()
