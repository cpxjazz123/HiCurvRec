"""
Task #71 — Exp A: 三个残差向量的形式对比

基于 Task #70 Scenario B 结论（d_H 是 d_E 的保序单调变换 → 最近邻等价），
本实验探索"等价性在量化内部是否成立"：
  r_E = r_1 - q          (欧氏减法)
  r_H = log_map(q_H, r_1_H)  (Poincaré 切空间向量)
  r_S = log_map(q_S, r_1_S)  (球面切空间向量)

评估指标：
  - norm mean / std / max
  - sparsity (% dims with |x| < 0.001)
  - PCA rank 95%
  - kurtosis
  - cosine similarity E-H / E-S / H-S

启动:
  python scripts/task6_exp_a_form.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


# ============== 流形几何函数 ==============

def project_to_poincare(x):
    """tanh(||x||) · x/||x|| 投影到 Poincaré 球"""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def project_to_sphere(x):
    """r/||r|| 投影到单位球"""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return x / norm


def poincare_log_map(q, r):
    """log_map at q, for r in Poincaré ball → returns tangent vector at q.

    Formula (Ganea 2018):
      λ_q = 2 / (1 - ||q||²)
      d_H(q, r) = arcosh(1 + 2||r-q||² / ((1-||q||²)(1-||r||²)))
      u = (r ⊕_c (-q)) where ⊕_c is Möbius addition
      u = ((1 + 2<q,r> + ||r||²) q + (1 - ||q||²) r) / (1 + 2<q,r> + ||q||²||r||²)
      log_q(r) = (d_H / λ_q) · u / ||u||
    """
    q_sqnorm = (q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    r_sqnorm = (r ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    inner = (q * r).sum(dim=-1, keepdim=True)

    # Möbius addition q ⊕ r
    num = (1 + 2 * inner + r_sqnorm) * q + (1 - q_sqnorm) * r
    denom = (1 + 2 * inner + q_sqnorm * r_sqnorm).clamp(min=1e-7)
    u = num / denom

    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-7)

    # Poincaré 距离
    diff_sq = ((r - q) ** 2).sum(dim=-1, keepdim=True)
    arg = (1 + 2 * diff_sq / ((1 - q_sqnorm) * (1 - r_sqnorm))).clamp(min=1 + 1e-7)
    d = torch.acosh(arg)

    # λ_q 共形因子
    lambda_q = 2 / (1 - q_sqnorm)

    return (d / (lambda_q * u_norm)) * u


def spherical_log_map(q, r):
    """log_map at q on unit sphere → tangent vector at q.

    Formula:
      d_S(q, r) = arccos(<q,r> / (||q|| ||r||))
      proj = r - <q,r> q  (projection onto tangent space at q)
      log_q(r) = (d_S / ||proj||) · proj
    """
    q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (q * r).sum(dim=-1, keepdim=True) / (q_norm * r_norm)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(cos_sim)

    inner = (q * r).sum(dim=-1, keepdim=True)
    proj = r - (inner / (q_norm ** 2)) * q
    proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=1e-7)

    return (d / proj_norm) * proj


# ============== 评估指标 ==============

def compute_metrics(residual, name):
    """对单个 (N, D) 残差张量计算所有形式指标"""
    res_np = residual.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(res_np, axis=-1)

    # Sparsity: % of |x| < 0.001
    sparsity = (np.abs(res_np) < 1e-3).mean()

    # Kurtosis (per dim, then mean) — use scipy.stats.kurtosis
    kurt = float(scipy_kurtosis(res_np, axis=0, fisher=True, nan_policy="omit").mean())

    # PCA rank 95%
    pca = PCA(n_components=min(256, res_np.shape[1]))
    pca.fit(res_np)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    return {
        "name": name,
        "shape": list(residual.shape),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_max": float(norms.max()),
        "norm_min": float(norms.min()),
        "sparsity_1e-3": float(sparsity),
        "kurtosis_mean": kurt,
        "pca_rank_95": rank_95,
    }


def cosine_sim_block(r1, r2):
    """计算 (r1, r2) 之间的逐样本 cosine sim 矩阵，返回 mean/std"""
    r1_norm = r1 / r1.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r2_norm = r2 / r2.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos = (r1_norm * r2_norm).sum(dim=-1)
    return float(cos.mean()), float(cos.std())


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqvae_ckpt", default="products/task16/from_task62/rqvae_ckpt.pt")
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_json", default="products/task16/from_task71/exp_a_form.json")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kmeans", action="store_true",
                        help="Train fresh MiniBatchKMeans instead of using Task #62 EMA RQ-VAE")
    args = parser.parse_args()

    print("=" * 60)
    print("Task #71 — Exp A: 三残差向量形式对比")
    print("=" * 60)

    # 加载 embeddings
    print(f"\n[1/4] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    if embeddings.dim() > 2:
        embeddings = embeddings.squeeze()
    print(f"  shape: {tuple(embeddings.shape)}, norm: {embeddings.norm(dim=-1).mean():.4f}")

    # Step 2: 训练/加载 RQ-VAE L1
    print(f"\n[2/4] Getting L1 codebook...")
    if args.use_kmeans:
        # 训练新鲜 MiniBatchKMeans，避免 Task #62 EMA artifact
        print(f"  Training fresh MiniBatchKMeans (n_clusters={args.n_clusters}, seed={args.seed})...")
        emb_np = embeddings.numpy()
        km = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            random_state=args.seed,
            batch_size=1024,
            n_init=3,
            max_iter=100,
        )
        km.fit(emb_np)
        q_idx = km.predict(emb_np)
        c_E = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        # 计算 L1 残差
        q = c_E[torch.tensor(q_idx)]
        r_1 = embeddings - q
        print(f"  Trained. Codebook norm: mean={c_E.norm(dim=-1).mean():.4f}, "
              f"min={c_E.norm(dim=-1).min():.4f}")
        print(f"  Unique clusters used: {len(set(q_idx.tolist()))}/{args.n_clusters}")
    else:
        # 用 Task #62 EMA RQ-VAE
        print(f"  Loading Task #62 EMA RQ-VAE: {args.rqvae_ckpt}")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from task62_train_rqvae import ResidualQuantization

        ckpt = torch.load(args.rqvae_ckpt, map_location="cpu", weights_only=False)
        model = ResidualQuantization(
            n_features=ckpt["n_features"],
            n_layers=ckpt["n_layers"],
            n_clusters=ckpt["n_clusters"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            _, _, residuals_list, _ = model(embeddings, update_codebook=False)
        r_1 = residuals_list[0]
        c_E = model.quantizers[0].codebook

    print(f"  r_1 shape: {tuple(r_1.shape)}, norm mean: {r_1.norm(dim=-1).mean():.4f}")
    print(f"  c_E shape: {tuple(c_E.shape)}")

    # 选最近码字（用 d_E，因为三种距离排序等价）
    d_E = torch.cdist(r_1, c_E, p=2)
    q_idx_t = d_E.argmin(dim=1)  # (N,)
    q = c_E[q_idx_t]  # (N, 768)
    print(f"  q shape: {tuple(q.shape)}")
    print(f"  unique codewords used: {len(set(q_idx_t.tolist()))}/{args.n_clusters}")

    # ===== 计算三个残差 =====
    # r_E = r_1 - q
    r_E = r_1 - q
    print(f"  r_E shape: {tuple(r_E.shape)}, norm mean: {r_E.norm(dim=-1).mean():.4f}")

    # r_H = log_map(q_H, r_1_H)
    r_1_H = project_to_poincare(r_1)
    q_H = project_to_poincare(q)
    r_H = poincare_log_map(q_H, r_1_H)
    print(f"  r_H shape: {tuple(r_H.shape)}, norm mean: {r_H.norm(dim=-1).mean():.4f}")

    # r_S = log_map(q_S, r_1_S)
    r_1_S = project_to_sphere(r_1)
    q_S = project_to_sphere(q)
    r_S = spherical_log_map(q_S, r_1_S)
    print(f"  r_S shape: {tuple(r_S.shape)}, norm mean: {r_S.norm(dim=-1).mean():.4f}")

    # ===== 计算形式指标 =====
    print(f"\n[4/4] Computing form metrics...")
    metrics_E = compute_metrics(r_E, "r_E (Euclidean)")
    metrics_H = compute_metrics(r_H, "r_H (Poincaré log_map)")
    metrics_S = compute_metrics(r_S, "r_S (Spherical log_map)")

    print(f"\n{'Metric':<20}{'r_E':<18}{'r_H':<18}{'r_S':<18}")
    print("-" * 74)
    for key in ["norm_mean", "norm_std", "norm_max", "norm_min",
                "sparsity_1e-3", "kurtosis_mean", "pca_rank_95"]:
        e = metrics_E[key]
        h = metrics_H[key]
        s = metrics_S[key]
        if isinstance(e, int):
            print(f"{key:<20}{e:<18}{h:<18}{s:<18}")
        else:
            print(f"{key:<20}{e:<18.6f}{h:<18.6f}{s:<18.6f}")

    # Cosine similarity (mean ± std)
    cos_EH = cosine_sim_block(r_E, r_H)
    cos_ES = cosine_sim_block(r_E, r_S)
    cos_HS = cosine_sim_block(r_H, r_S)
    print(f"\ncos(r_E, r_H) mean={cos_EH[0]:.6f} std={cos_EH[1]:.6f}")
    print(f"cos(r_E, r_S) mean={cos_ES[0]:.6f} std={cos_ES[1]:.6f}")
    print(f"cos(r_H, r_S) mean={cos_HS[0]:.6f} std={cos_HS[1]:.6f}")

    # 决策
    print(f"\n{'='*60}")
    print("Decision (R1: 三残差是否本质不同?)")
    print(f"{'='*60}")
    r1_confirmed = (cos_EH[0] < 0.95) and (cos_ES[0] < 0.95)
    if r1_confirmed:
        decision = "R1_CONFIRMED (三残差本质不同 → 继续 B/C/D)"
    else:
        decision = "R1_REJECTED (cos(E,H) 或 cos(E,S) ≥ 0.95 → STOP)"

    print(f"  Decision: {decision}")

    # 保存
    out = {
        "task": "#71 Exp A — form comparison",
        "n_items": int(embeddings.shape[0]),
        "metrics": {
            "r_E": metrics_E,
            "r_H": metrics_H,
            "r_S": metrics_S,
        },
        "cosine_sim": {
            "E_H": {"mean": cos_EH[0], "std": cos_EH[1]},
            "E_S": {"mean": cos_ES[0], "std": cos_ES[1]},
            "H_S": {"mean": cos_HS[0], "std": cos_HS[1]},
        },
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()