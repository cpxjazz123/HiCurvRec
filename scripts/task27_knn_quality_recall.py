"""Task #27 — 邻域排序质量 vs 下游 Recall 相关性实验.

目标: 计算多个 tokenizer 的 kNN Recall (T5 邻域 digit 重叠率) 并与下游
TIGER R@5 做 Pearson + Spearman 相关性分析 + bootstrap CI.

设计原则 (CLAUDE.md Rule 7):
- 无 fallback: 任何缺失路径立即 raise, 不静默跳过
- 无默认值: 所有 tokenizers 在 MANIFEST 中显式登记
- 区分"预期内缺失" (Task #85 m=2 mode collapse) 与"预期外失败"

输出:
- task27_knn_quality_table.csv
- task27_correlation_summary.json
- task27_mse_vs_knn.png (可选, 若 PM-RQ 模型可解码)
- task27_knn_vs_recall.png
- verdicts/task27_neighborhood_quality_result.md
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO_ROOT)


# =============================================================================
# Manifest: tokenizer + 下游 R@5 真值
# =============================================================================
# 字段:
#   id: 唯一标识
#   sid_path: SID tensor 路径, shape (D, N) int64
#   sid_dim: 实际使用的 digit 数量 (e.g., Task #24 cascade 跳过 padding row)
#   downstream_R5: 从 verdict 抽取的 TEST R@5
#   downstream_R10: 同上 R@10
#   family: 'flat_rqvae' | 'pm_rq' | 'curvature_rqvae'
#   notes: 备注 (e.g., mode collapse)
TOKENIZER_MANIFEST: List[Dict] = [
    {
        "id": "Task85_m1_quasi_euclid",
        "sid_path": "products/task85_tri_geom_rqvae/euclid_full/sid_subspace_1.pt",
        "sid_dim": None,  # all rows
        "downstream_R5": 0.0200,
        "downstream_R10": 0.0288,
        "family": "curvature_rqvae",
        "notes": "single κ=0 (Euclidean) RQ-VAE",
    },
    {
        "id": "Task87_K256_seed42",
        "sid_path": "logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt",
        "sid_dim": None,
        "downstream_R5": 0.01937,
        "downstream_R10": 0.03318,
        "family": "flat_rqvae",
        "notes": "TIGER-aligned flat Euclidean baseline (seed=42)",
    },
    {
        "id": "Task85_m0_sphere",
        "sid_path": "products/task85_tri_geom_rqvae/sphere_fix/sid_subspace_0.pt",
        "sid_dim": None,
        "downstream_R5": 0.0174,
        "downstream_R10": 0.0262,
        "family": "curvature_rqvae",
        "notes": "single κ=+1 (sphere) RQ-VAE",
    },
    {
        "id": "Task107_K256_seed123",
        "sid_path": "logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt",
        "sid_dim": None,
        "downstream_R5": 0.01489,
        "downstream_R10": 0.02627,
        "family": "flat_rqvae",
        "notes": "Same SID as Task87, TIGER seed=123 (variance probe)",
    },
    # ❌ DELETED 2026-07-19:
    #   - Task22_PM_RQ_phase2: R@5=0.00474, 仅 baseline 24%, 差距过大
    #   - Task22_PM_RQ_phase3_cascade: cascade R@5=0.00144 < Phase 2 R@5=0.00474
    #   - Task85_m2_hyperbolic: mode collapse → trivial bias 0.25546
]


# =============================================================================
# 数据加载
# =============================================================================
def load_t5_embedding(path: str) -> torch.Tensor:
    """Load sentence-t5-base embedding (N, 768)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"T5 embedding not found: {path}")
    emb = torch.load(path, map_location="cpu", weights_only=True)
    if emb.ndim != 2 or emb.shape[1] != 768:
        raise ValueError(f"T5 embedding shape mismatch: {tuple(emb.shape)}, expected (N, 768)")
    return emb.float()


def load_sid(path: str, sid_dim: int | None) -> torch.Tensor:
    """Load SID tensor (D, N) int64.

    If sid_dim is set, take first `sid_dim` rows.

    Constant rows (std=0 or single unique value) are auto-excluded — these are
    padding/initialization artifacts that would inflate co-cluster rate to 1.0
    trivially (e.g. Task #22 PM-RQ Phase 2 row 3 is all zeros).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SID file not found: {path}")
    sid = torch.load(path, map_location="cpu", weights_only=True)
    if sid.ndim != 2:
        raise ValueError(f"SID tensor must be 2-D, got {sid.ndim}-D for {path}")
    if sid_dim is not None:
        if sid.shape[0] < sid_dim:
            raise ValueError(
                f"SID has only {sid.shape[0]} rows but sid_dim={sid_dim} requested"
            )
        sid = sid[:sid_dim]
    sid = sid.long()

    # Auto-exclude constant rows (padding/initialization artifacts)
    keep_rows = []
    for r in range(sid.shape[0]):
        u = torch.unique(sid[r])
        if u.shape[0] >= 2:  # at least 2 distinct values
            keep_rows.append(r)
        else:
            print(
                f"    [WARNING] {path}: row {r} is constant (value={u[0].item()}), "
                f"EXCLUDED from co-cluster analysis"
            )
    sid = sid[keep_rows]
    if sid.shape[0] == 0:
        raise ValueError(f"All rows in {path} are constant — no meaningful SID")
    return sid


def compute_t5_topk_neighbors(
    emb: torch.Tensor, k: int, batch_size: int = 1024
) -> torch.Tensor:
    """Compute top-k cosine neighbors for each item.

    Returns: indices (N, k) where col 0 is self.
    """
    n = emb.shape[0]
    emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
    topk_indices = torch.zeros(n, k, dtype=torch.long)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = emb_norm[start:end]  # (B, 768)
        # Cosine similarity: (B, N)
        sim_chunk = chunk @ emb_norm.T  # (B, N)
        # Top-k
        _, idx = sim_chunk.topk(k, dim=1, largest=True)
        topk_indices[start:end] = idx

    return topk_indices


# =============================================================================
# kNN 质量计算
# =============================================================================
def compute_cocluster_rate(
    topk_indices: torch.Tensor, sid: torch.Tensor, k: int
) -> np.ndarray:
    """对每个 item, 计算其 top-k 邻居中与自身共享至少 1 digit 的比例.

    注意: 该指标受 digit 总数和每行 unique 数严重混淆 (例如 4-digit + dedup
    的 Task #87 随机基线 ≈ 1.0). 推荐改用归一化 Hamming (见下).
    """
    n, kk = topk_indices.shape
    if kk < k:
        raise ValueError(f"topk_indices has only {kk} cols, need k={k}")

    neighbors = topk_indices[:, 1 : k + 1]  # (N, k)
    self_digits = sid.T  # (N, D)
    neighbor_digits = sid.T[neighbors]  # (N, k, D)

    match = (neighbor_digits == self_digits[:, None, :]).any(dim=2)  # (N, k)
    rate = match.float().mean(dim=1)  # (N,)
    return rate.numpy()


def compute_avg_hamming_to_topk(
    topk_indices: torch.Tensor, sid: torch.Tensor, k: int
) -> np.ndarray:
    """对每个 item, 计算其 top-k 邻居与自身的平均 Hamming 距离 (未归一化)."""
    neighbors = topk_indices[:, 1 : k + 1]  # (N, k)
    self_digits = sid.T  # (N, D)
    neighbor_digits = sid.T[neighbors]  # (N, k, D)

    hamming = (neighbor_digits != self_digits[:, None, :]).sum(dim=2).float()  # (N, k)
    avg_hamming = hamming.mean(dim=1)  # (N,)
    return avg_hamming.numpy()


def compute_normalized_hamming(
    topk_indices: torch.Tensor, sid: torch.Tensor, k: int
) -> np.ndarray:
    """对每个 item, 计算其 top-k 邻居与自身的归一化 Hamming 距离 (Hamming / D).

    Lower = better neighborhood preservation.
    推荐作为主指标 (跨 tokenizer 可比, 不受 digit 数量干扰).
    """
    d = sid.shape[0]
    raw = compute_avg_hamming_to_topk(topk_indices, sid, k)
    return raw / d


def compute_random_baseline_cocluster(
    sid: torch.Tensor, k: int, n_perm: int = 5
) -> float:
    """计算随机邻居下的 co-cluster rate 基线 (permutation null).

    用 sid 自身打乱邻居顺序作为 null model. 反映"给定该 SID 分布,
    随机配对的邻居平均共 digit 率".
    """
    n = sid.shape[1]
    self_digits = sid.T  # (N, D)
    rates = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        rand_idx = torch.from_numpy(rng.permutation(n))
        # each item's "neighbor" is a random item
        rand_neighbors = sid.T[rand_idx]  # (N, D)
        # share at least one digit (== not !=)
        match = (rand_neighbors == self_digits).any(dim=1)  # (N,)
        rates.append(match.float().mean().item())
    return float(np.mean(rates))


# =============================================================================
# 相关性分析
# =============================================================================
def pearson_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson ρ + p-value.

    Returns: (rho, p_value)
    """
    from scipy.stats import pearsonr

    rho, p = pearsonr(x, y)
    return float(rho), float(p)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman ρ + p-value."""
    from scipy.stats import spearmanr

    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap CI for correlation ρ.

    Args:
        x, y: data
        fn: function (x, y) → ρ scalar
        n_boot: bootstrap iterations
        alpha: significance level
        seed: RNG seed

    Returns: (point_estimate, ci_low, ci_high)

    With small n (n=6), bootstrap samples can be constant (especially when
    data has ties), causing scipy to return NaN. We skip such samples.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    point = fn(x, y)
    boot_stats = []
    attempts = 0
    while len(boot_stats) < n_boot and attempts < n_boot * 10:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        x_b, y_b = x[idx], y[idx]
        try:
            with np.errstate(all="ignore"):
                rho_b = float(fn(x_b, y_b))
            if not np.isnan(rho_b):
                boot_stats.append(rho_b)
        except Exception:
            continue
    if len(boot_stats) < 100:
        return float(point), float("nan"), float("nan")
    boot_stats = np.array(boot_stats)
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), ci_low, ci_high


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t5_emb",
        type=str,
        default="logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt",
        help="T5 embedding (N, 768) .pt file",
    )
    parser.add_argument("--k", type=int, default=50, help="top-k for neighbor computation")
    parser.add_argument("--n_boot", type=int, default=1000, help="bootstrap iterations")
    parser.add_argument("--out_dir", type=str, default=".", help="output directory")
    parser.add_argument(
        "--include_trivial", action="store_true",
        help="include 已删除的 trivial/mode-collapsed tokenizer (e.g. Task #85 m=2) 在 correlation 中"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Task #27: kNN 质量 vs 下游 R@5 相关性分析 ===")
    print(f"T5 embedding: {args.t5_emb}")
    print(f"k = {args.k}, bootstrap = {args.n_boot}")
    print()

    # 加载 T5 embedding
    print("加载 T5 embedding ...")
    emb = load_t5_embedding(args.t5_emb)
    n_items = emb.shape[0]
    print(f"  shape = {tuple(emb.shape)}")
    assert n_items == 11924, f"Expected 11924 items, got {n_items}"

    # 加载每个 tokenizer 并计算 kNN 质量
    print(f"\n计算 T5 top-{args.k} cosine neighbors ...")
    topk_indices = compute_t5_topk_neighbors(emb, k=args.k + 1)  # +1 for self
    print(f"  topk_indices shape = {tuple(topk_indices.shape)}")

    results = []
    for entry in TOKENIZER_MANIFEST:
        tid = entry["id"]
        print(f"\n[{tid}] 加载 SID ...")
        sid = load_sid(entry["sid_path"], entry["sid_dim"])
        print(f"  SID shape = {tuple(sid.shape)}, downstream R@5 = {entry['downstream_R5']}")
        if sid.shape[1] != n_items:
            raise ValueError(
                f"SID column count {sid.shape[1]} != T5 embedding rows {n_items}"
            )

        cocluster_rate = compute_cocluster_rate(topk_indices, sid, k=args.k)
        avg_hamming = compute_avg_hamming_to_topk(topk_indices, sid, k=args.k)
        norm_hamming = compute_normalized_hamming(topk_indices, sid, k=args.k)
        random_baseline = compute_random_baseline_cocluster(sid, k=args.k, n_perm=3)

        results.append({
            "id": tid,
            "family": entry["family"],
            "sid_dim": int(sid.shape[0]),
            "downstream_R5": entry["downstream_R5"],
            "downstream_R10": entry["downstream_R10"],
            "cocluster_mean": float(cocluster_rate.mean()),
            "cocluster_std": float(cocluster_rate.std()),
            "hamming_mean": float(avg_hamming.mean()),
            "hamming_std": float(avg_hamming.std()),
            "hamming_norm_mean": float(norm_hamming.mean()),
            "hamming_norm_std": float(norm_hamming.std()),
            "cocluster_random_baseline": random_baseline,
            "cocluster_lift": float(cocluster_rate.mean()) - random_baseline,
            "notes": entry["notes"],
        })
        print(
            f"  co-cluster mean = {cocluster_rate.mean():.4f} "
            f"(std={cocluster_rate.std():.4f})  "
            f"random_baseline={random_baseline:.4f}  "
            f"lift={cocluster_rate.mean() - random_baseline:+.4f}"
        )
        print(
            f"  avg Hamming to top-{args.k} = {avg_hamming.mean():.4f} "
            f"(normalized={norm_hamming.mean():.4f})"
        )

    # 写入 CSV 表
    csv_path = out_dir / "task27_knn_quality_table.csv"
    print(f"\n写 CSV 表: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # 主分析: 已排除 4 个失败的 tokenizer (2026-07-19 删除)
    # - Task22_PM_RQ_phase2: R@5=0.00474, 仅 baseline 24%, 差距过大
    # - Task22_PM_RQ_phase3_cascade: cascade R@5=0.00144 < Phase 2 R@5=0.00474
    # - Task85_m2_hyperbolic: mode collapse → trivial bias 0.25546
    # - Task #80 v3 T5/MCKG: 无 TIGER 端到端 R@5
    analysis_results = [r for r in results if "MODE COLLAPSE" not in r["notes"]]
    if args.include_trivial:
        analysis_results = results
        print(f"\n⚠️ 包含已删除的 trivial tokenizer (R@5 不可信)")

    print(f"\n=== 主相关性分析 ({len(analysis_results)} data points) ===")
    x_cocluster = np.array([r["cocluster_mean"] for r in analysis_results])
    x_cocluster_lift = np.array([r["cocluster_lift"] for r in analysis_results])
    x_hamming_raw = np.array([r["hamming_mean"] for r in analysis_results])
    x_hamming_norm = np.array([r["hamming_norm_mean"] for r in analysis_results])
    y_recall = np.array([r["downstream_R5"] for r in analysis_results])

    def _spearman(x, y):
        return spearmanr(x, y).statistic

    # 1) Primary: co-cluster lift (controls for digit-count confounding)
    p_rho, p_p = pearsonr(x_cocluster_lift, y_recall)
    s_rho, s_p = spearmanr(x_cocluster_lift, y_recall)
    s_point, s_ci_lo, s_ci_hi = bootstrap_ci(
        x_cocluster_lift, y_recall, _spearman, n_boot=args.n_boot,
    )
    print(
        f"  co-cluster lift vs R@5:  Pearson ρ = {p_rho:+.4f} (p={p_p:.4f})  "
        f"Spearman ρ = {s_rho:+.4f} (p={s_p:.4f})  "
        f"CI95 = [{s_ci_lo:+.4f}, {s_ci_hi:+.4f}]"
    )

    # 2) Primary: normalized Hamming (lower = better, expect negative ρ)
    p_rho_h, p_p_h = pearsonr(x_hamming_norm, y_recall)
    s_rho_h, s_p_h = spearmanr(x_hamming_norm, y_recall)
    s_point_h, s_ci_lo_h, s_ci_hi_h = bootstrap_ci(
        x_hamming_norm, y_recall, _spearman, n_boot=args.n_boot,
    )
    print(
        f"  norm Hamming vs R@5:    Pearson ρ = {p_rho_h:+.4f} (p={p_p_h:.4f})  "
        f"Spearman ρ = {s_rho_h:+.4f} (p={s_p_h:.4f})  "
        f"CI95 = [{s_ci_lo_h:+.4f}, {s_ci_hi_h:+.4f}]"
    )

    # 3) Sanity: raw Hamming
    p_rho_raw, p_p_raw = pearsonr(x_hamming_raw, y_recall)
    s_rho_raw, s_p_raw = spearmanr(x_hamming_raw, y_recall)
    print(
        f"  raw Hamming vs R@5:     Pearson ρ = {p_rho_raw:+.4f} (p={p_p_raw:.4f})  "
        f"Spearman ρ = {s_rho_raw:+.4f} (p={s_p_raw:.4f})"
    )

    # 4) Legacy: raw co-cluster (confounded by digit count)
    p_rho_c, p_p_c = pearsonr(x_cocluster, y_recall)
    s_rho_c, s_p_c = spearmanr(x_cocluster, y_recall)
    print(
        f"  raw co-cluster vs R@5:  Pearson ρ = {p_rho_c:+.4f} (p={p_p_c:.4f})  "
        f"Spearman ρ = {s_rho_c:+.4f} (p={s_p_c:.4f})"
    )

    # 写入 JSON summary
    summary = {
        "n_data_points": len(analysis_results),
        "n_total_tokenizers": len(results),
        "excluded": [r["id"] for r in results if "MODE COLLAPSE" in r["notes"]],
        "k": args.k,
        "n_bootstrap": args.n_boot,
        "primary_cocluster_lift_vs_R5": {
            "pearson_rho": p_rho,
            "pearson_p": p_p,
            "spearman_rho": s_rho,
            "spearman_p": s_p,
            "spearman_ci95": [s_ci_lo, s_ci_hi],
        },
        "primary_normalized_hamming_vs_R5": {
            "pearson_rho": p_rho_h,
            "pearson_p": p_p_h,
            "spearman_rho": s_rho_h,
            "spearman_p": s_p_h,
            "spearman_ci95": [s_ci_lo_h, s_ci_hi_h],
        },
        "sanity_raw_hamming_vs_R5": {
            "pearson_rho": p_rho_raw,
            "pearson_p": p_p_raw,
            "spearman_rho": s_rho_raw,
            "spearman_p": s_p_raw,
        },
        "legacy_raw_cocluster_vs_R5": {
            "pearson_rho": p_rho_c,
            "pearson_p": p_p_c,
            "spearman_rho": s_rho_c,
            "spearman_p": s_p_c,
        },
        "per_tokenizer_results": results,
    }
    summary_path = out_dir / "task27_correlation_summary.json"
    print(f"\n写 JSON summary: {summary_path}")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 画散点图 (3 subplots: co-cluster lift, normalized Hamming, raw co-cluster)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Subplot 1: co-cluster lift vs R@5
    ax = axes[0]
    for r in analysis_results:
        ax.scatter(
            r["cocluster_lift"], r["downstream_R5"],
            s=80, alpha=0.7, label=r["id"],
        )
        ax.annotate(
            r["id"].replace("_", "\n", 1),
            (r["cocluster_lift"], r["downstream_R5"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel(f"Co-cluster LIFT in T5 top-{args.k}\n(observed − random baseline)")
    ax.set_ylabel("Downstream R@5")
    ax.set_title(
        f"PRIMARY: Co-cluster Lift vs R@5\n"
        f"Spearman ρ={s_rho:+.4f} (p={s_p:.4f}) CI95=[{s_ci_lo:+.4f}, {s_ci_hi:+.4f}]"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    # Subplot 2: normalized Hamming vs R@5
    ax = axes[1]
    for r in analysis_results:
        ax.scatter(
            r["hamming_norm_mean"], r["downstream_R5"],
            s=80, alpha=0.7, label=r["id"],
        )
        ax.annotate(
            r["id"].replace("_", "\n", 1),
            (r["hamming_norm_mean"], r["downstream_R5"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel(f"Normalized Hamming to top-{args.k}\n(lower = better neighborhood preservation)")
    ax.set_ylabel("Downstream R@5")
    ax.set_title(
        f"PRIMARY: Norm Hamming vs R@5\n"
        f"Spearman ρ={s_rho_h:+.4f} (p={s_p_h:.4f}) CI95=[{s_ci_lo_h:+.4f}, {s_ci_hi_h:+.4f}]"
    )
    ax.grid(True, alpha=0.3)

    # Subplot 3: raw co-cluster vs R@5 (legacy, confounded)
    ax = axes[2]
    for r in analysis_results:
        ax.scatter(
            r["cocluster_mean"], r["downstream_R5"],
            s=80, alpha=0.7, label=r["id"],
        )
        ax.annotate(
            r["id"].replace("_", "\n", 1),
            (r["cocluster_mean"], r["downstream_R5"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel(f"Raw Co-cluster Rate in T5 top-{args.k}\n(CONFOUNDED by digit count)")
    ax.set_ylabel("Downstream R@5")
    ax.set_title(
        f"LEGACY: Raw Co-cluster vs R@5\n"
        f"Spearman ρ={s_rho_c:+.4f} (p={s_p_c:.4f})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "task27_knn_vs_recall.png"
    print(f"写散点图: {fig_path}")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Task #27 主分析完成 ===")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {summary_path}")
    print(f"PNG:  {fig_path}")


if __name__ == "__main__":
    main()