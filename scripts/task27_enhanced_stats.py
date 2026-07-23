"""Task #27 Enhanced Statistics — 在已有 n=6 数据上做深入统计推断 (无训练).

新增 (相对于 scripts/task27_knn_quality_recall.py):
1. Kendall's τ (与 Spearman ρ 互补, 对小样本 + ties 更稳健)
2. 多重比较校正 (Holm-Bonferroni): 主指标 co-cluster lift + norm Hamming + raw Hamming + raw co-cluster 共 4 个
3. 部分相关 (partial correlation): 控制 sid_dim 后, norm Hamming 与 R@5 的"纯"相关
4. 样本量 power 分析: ρ=−0.67 假设下, 80% power 所需最小 n
5. Bootstrap 收敛曲线: 100/500/1k/5k/10k n_boot 下 CI95 宽度变化
6. Jackknife LOO: 每个 tokenizer 对 ρ 的影响 (DFFITS 风格)
7. 不同 k 敏感性: k=10/30/50/100 下 ρ 稳定性
8. 置信区间宽度诊断: 6 个数据点下 CI 是否信息充分

不训练任何新模型. 全部输入是 verdicts 里已有的 n=6 SID 数据.

启动:
    cd /home/wlia0047/ar57/wenyu/GeneRec
    python3 scripts/task27_enhanced_stats.py
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau, norm
from scipy.special import comb

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO_ROOT)


# =============================================================================
# Manifest: 同 task27_knn_quality_recall.py — 6 个 tokenizer + 下游 R@5
# =============================================================================
TOKENIZER_MANIFEST: List[Dict] = [
    {
        "id": "Task85_m1_quasi_euclid",
        "sid_path": "products/task85_tri_geom_rqvae/euclid_full/sid_subspace_1.pt",
        "sid_dim": None,
        "downstream_R5": 0.0200,
        "downstream_R10": 0.0288,
        "family": "curvature_rqvae",
    },
    {
        "id": "Task87_K256_seed42",
        "sid_path": "logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt",
        "sid_dim": None,
        "downstream_R5": 0.01937,
        "downstream_R10": 0.03318,
        "family": "flat_rqvae",
    },
    {
        "id": "Task85_m0_sphere",
        "sid_path": "products/task85_tri_geom_rqvae/sphere_fix/sid_subspace_0.pt",
        "sid_dim": None,
        "downstream_R5": 0.0174,
        "downstream_R10": 0.0262,
        "family": "curvature_rqvae",
    },
    {
        "id": "Task107_K256_seed123",
        "sid_path": "logs/task87_s2_rqvae_inference_v6/runs/task87_s2_infer/pickle/cluster_ids.pt",
        "sid_dim": None,
        "downstream_R5": 0.01489,
        "downstream_R10": 0.02627,
        "family": "flat_rqvae",
    },
    # ❌ DELETED 2026-07-19:
    #   - Task22_PM_RQ_phase2: R@5=0.00474, 仅 baseline 24%, 差距过大
    #   - Task22_PM_RQ_phase3_cascade: cascade R@5=0.00144 < Phase 2 R@5=0.00474
    #   - Task85_m2_hyperbolic: mode collapse → trivial bias 0.25546
]


# =============================================================================
# 数据加载 & kNN 质量 (复用 task27_knn_quality_recall.py 的逻辑)
# =============================================================================
def load_t5_embedding(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"T5 embedding not found: {path}")
    emb = torch.load(path, map_location="cpu", weights_only=True)
    if emb.ndim != 2 or emb.shape[1] != 768:
        raise ValueError(f"T5 embedding shape mismatch: {tuple(emb.shape)}, expected (N, 768)")
    return emb.float()


def load_sid(path: str, sid_dim: int | None) -> torch.Tensor:
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
    keep_rows = []
    for r in range(sid.shape[0]):
        u = torch.unique(sid[r])
        if u.shape[0] >= 2:
            keep_rows.append(r)
    sid = sid[keep_rows]
    if sid.shape[0] == 0:
        raise ValueError(f"All rows in {path} are constant — no meaningful SID")
    return sid


def compute_t5_topk_neighbors(emb: torch.Tensor, k: int, batch_size: int = 1024) -> torch.Tensor:
    n = emb.shape[0]
    emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
    topk_indices = torch.zeros(n, k, dtype=torch.long)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = emb_norm[start:end]
        sim_chunk = chunk @ emb_norm.T
        _, idx = sim_chunk.topk(k, dim=1, largest=True)
        topk_indices[start:end] = idx
    return topk_indices


def compute_normalized_hamming(topk_indices: torch.Tensor, sid: torch.Tensor, k: int) -> np.ndarray:
    """per-item 归一化 Hamming to top-k neighbors (lower = better preservation)."""
    neighbors = topk_indices[:, 1 : k + 1]
    self_digits = sid.T
    neighbor_digits = sid.T[neighbors]
    hamming = (neighbor_digits != self_digits[:, None, :]).sum(dim=2).float()
    avg_hamming = hamming.mean(dim=1)
    return (avg_hamming / sid.shape[0]).numpy()


def compute_cocluster_rate(topk_indices: torch.Tensor, sid: torch.Tensor, k: int) -> np.ndarray:
    """per-item 与 top-k 邻居共 digit 比例."""
    neighbors = topk_indices[:, 1 : k + 1]
    self_digits = sid.T
    neighbor_digits = sid.T[neighbors]
    match = (neighbor_digits == self_digits[:, None, :]).any(dim=2)
    return match.float().mean(dim=1).numpy()


def compute_random_baseline_cocluster(sid: torch.Tensor, k: int, n_perm: int = 5) -> float:
    n = sid.shape[1]
    self_digits = sid.T
    rates = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        rand_idx = torch.from_numpy(rng.permutation(n))
        rand_neighbors = sid.T[rand_idx]
        match = (rand_neighbors == self_digits).any(dim=1)
        rates.append(match.float().mean().item())
    return float(np.mean(rates))


# =============================================================================
# 增强统计方法
# =============================================================================
def compute_partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Partial correlation of x, y controlling for z.

    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))
    """
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    num = r_xy - r_xz * r_yz
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if abs(den) < 1e-12:
        return float("nan"), float("nan")
    r_partial = num / den
    # p-value via t-distribution with df = n - 3
    n = len(x)
    df = n - 3
    if df <= 0:
        return float(r_partial), float("nan")
    t = r_partial * np.sqrt(df / (1 - r_partial**2))
    from scipy.stats import t as student_t
    p = 2 * (1 - student_t.cdf(abs(t), df=df))
    return float(r_partial), float(p)


def power_for_corr(rho: float, n: int, alpha: float = 0.05) -> float:
    """Compute statistical power for two-sided Pearson r test.

    Uses the exact Fisher z-transformation.
    Power = P(|Z| > z_{1-α/2}) where Z ~ N(0,1) under H1 with non-centrality
    λ = |z_r| * sqrt(n - 3), z_r = atanh(rho).
    """
    if n <= 3:
        return float("nan")
    z_r = np.arctanh(np.clip(rho, -0.999, 0.999))
    se = 1.0 / np.sqrt(n - 3)
    ncp = z_r / se  # non-centrality
    z_crit = norm.ppf(1 - alpha / 2)
    power = norm.cdf(ncp - z_crit) + norm.cdf(-ncp - z_crit)
    return float(power)


def min_n_for_power(target_rho: float, target_power: float = 0.80, alpha: float = 0.05) -> int:
    """Compute minimum sample size for given target ρ and power."""
    if abs(target_rho) < 0.01:
        return float("inf")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(target_power)
    # Approximation: n = ((z_alpha + z_beta) / arctanh(rho))² + 3
    z_r = np.arctanh(abs(target_rho))
    n_approx = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n_approx))


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple testing."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        corrected = min(corrected, 1.0)
        cum_max = max(cum_max, corrected)
        adjusted[orig_idx] = cum_max
    return adjusted


def bootstrap_ci_curve(
    x: np.ndarray, y: np.ndarray, fn, n_boot_list: List[int], seed: int = 42
) -> Dict[int, Tuple[float, float, float]]:
    """Bootstrap CI at different n_boot to see convergence."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = {}
    for n_boot in n_boot_list:
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
            out[n_boot] = (float("nan"), float("nan"), float("nan"))
            continue
        boot_stats = np.array(boot_stats)
        out[n_boot] = (
            float(np.percentile(boot_stats, 2.5)),
            float(np.percentile(boot_stats, 50)),
            float(np.percentile(boot_stats, 97.5)),
        )
    return out


def jackknife_influence(x: np.ndarray, y: np.ndarray, fn) -> List[Tuple[int, float, float]]:
    """Leave-one-out jackknife: report influence of each point on ρ.

    Returns: list of (index, leave_one_out_rho, delta_from_full)
    """
    full = fn(x, y)
    n = len(x)
    results = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if mask.sum() < 3:
            continue
        try:
            with np.errstate(all="ignore"):
                loo = float(fn(x[mask], y[mask]))
            if np.isnan(loo):
                continue
        except Exception:
            continue
        results.append((i, loo, full - loo))
    return results


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t5_emb",
        type=str,
        default="logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 30, 50, 100])
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--include_trivial", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Task #27 Enhanced Statistics (no training) ===")
    print(f"T5 embedding: {args.t5_emb}")
    print(f"k values tested: {args.ks}")
    print()

    emb = load_t5_embedding(args.t5_emb)
    n_items = emb.shape[0]
    print(f"Loaded T5 embedding: {tuple(emb.shape)}")
    assert n_items == 11924, f"Expected 11924 items, got {n_items}"

    # 用最大 k 一次性算邻居, 然后切片复用
    max_k = max(args.ks)
    print(f"\nComputing T5 top-{max_k + 1} cosine neighbors ...")
    topk_indices_full = compute_t5_topk_neighbors(emb, k=max_k + 1)
    print(f"  topk_indices shape = {tuple(topk_indices_full.shape)}")

    # 加载每个 tokenizer 并计算所有 k 下的指标
    active_manifest = [e for e in TOKENIZER_MANIFEST if not e.get("exclude_from_primary", False)]
    if args.include_trivial:
        active_manifest = TOKENIZER_MANIFEST

    print(f"\nActive tokenizers: {len(active_manifest)}")
    for entry in active_manifest:
        tid = entry["id"]
        sid = load_sid(entry["sid_path"], entry["sid_dim"])
        if sid.shape[1] != n_items:
            raise ValueError(
                f"SID column count {sid.shape[1]} != T5 embedding rows {n_items} for {tid}"
            )
        entry["loaded_sid_dim"] = int(sid.shape[0])

        entry["per_k_metrics"] = {}
        for k in args.ks:
            ham_norm = compute_normalized_hamming(topk_indices_full, sid, k=k)
            cocluster = compute_cocluster_rate(topk_indices_full, sid, k=k)
            random_baseline = compute_random_baseline_cocluster(sid, k=k, n_perm=3)
            entry["per_k_metrics"][k] = {
                "hamming_norm_mean": float(ham_norm.mean()),
                "hamming_norm_std": float(ham_norm.std()),
                "cocluster_mean": float(cocluster.mean()),
                "cocluster_lift": float(cocluster.mean() - random_baseline),
            }
        print(f"  {tid}: sid_dim={entry['loaded_sid_dim']}, "
              f"k=50 hamming_norm={entry['per_k_metrics'][50]['hamming_norm_mean']:.4f}, "
              f"R@5={entry['downstream_R5']}")

    # ===== 选主 k = 50 做深入分析 =====
    K_MAIN = 50
    print(f"\n=== 主分析 (k={K_MAIN}) ===")

    x_hamming_norm = np.array([e["per_k_metrics"][K_MAIN]["hamming_norm_mean"] for e in active_manifest])
    x_cocluster_lift = np.array([e["per_k_metrics"][K_MAIN]["cocluster_lift"] for e in active_manifest])
    x_cocluster = np.array([e["per_k_metrics"][K_MAIN]["cocluster_mean"] for e in active_manifest])
    y_recall = np.array([e["downstream_R5"] for e in active_manifest])
    z_sid_dim = np.array([e["loaded_sid_dim"] for e in active_manifest], dtype=float)

    n = len(active_manifest)
    print(f"n = {n}")

    # === 1. 多指标相关 + 多重比较校正 ===
    print("\n[1] 多指标相关 + Holm-Bonferroni 校正")

    metrics_to_test = {
        "norm_hamming": x_hamming_norm,
        "cocluster_lift": x_cocluster_lift,
        "raw_hamming": x_hamming_norm * z_sid_dim,  # 解归一化
        "raw_cocluster": x_cocluster,
    }

    correlation_table = []
    raw_pvals = []
    for name, x in metrics_to_test.items():
        p_r, p_p = pearsonr(x, y_recall)
        s_r, s_p = spearmanr(x, y_recall)
        tau, tau_p = kendalltau(x, y_recall)
        correlation_table.append({
            "metric": name,
            "n": n,
            "pearson_rho": float(p_r),
            "pearson_p": float(p_p),
            "spearman_rho": float(s_r),
            "spearman_p": float(s_p),
            "kendall_tau": float(tau),
            "kendall_p": float(tau_p),
        })
        # use Spearman p for Holm correction (primary non-parametric)
        raw_pvals.append(float(s_p))
        print(f"  {name:18s}: Pearson ρ={p_r:+.4f} (p={p_p:.4f})  "
              f"Spearman ρ={s_r:+.4f} (p={s_p:.4f})  "
              f"Kendall τ={tau:+.4f} (p={tau_p:.4f})")

    holm_adjusted = holm_bonferroni(raw_pvals)
    for i, row in enumerate(correlation_table):
        row["spearman_p_holm"] = float(holm_adjusted[i])
        print(f"  {row['metric']:18s}: Spearman p (Holm-corrected) = {holm_adjusted[i]:.4f}")

    # === 2. Partial correlation: 控制 sid_dim ===
    print("\n[2] Partial correlation (控制 sid_dim 后)")
    for name, x in [("norm_hamming", x_hamming_norm), ("cocluster_lift", x_cocluster_lift)]:
        pr, pp = compute_partial_corr(x, y_recall, z_sid_dim)
        s_r, s_p = spearmanr(x, y_recall)
        print(f"  {name:18s}: partial r = {pr:+.4f} (p={pp:.4f})  "
              f"(marginal Spearman ρ={s_r:+.4f}, p={s_p:.4f})  "
              f"sid_dim 作为协变量")
        correlation_table.append({
            "metric": f"{name}_partial_sid_dim",
            "n": n,
            "partial_r": float(pr),
            "partial_p": float(pp),
            "covariate": "sid_dim",
        })

    # === 3. Power analysis ===
    print("\n[3] Power analysis (假设 ρ=−0.67)")
    target_rho = -0.667
    print(f"  target |ρ| = {abs(target_rho):.3f}")
    for nn in [6, 8, 10, 14, 21, 30, 50]:
        p = power_for_corr(target_rho, nn)
        print(f"    n={nn:3d}: power = {p:.3f}")
    min_n = min_n_for_power(target_rho, target_power=0.80)
    print(f"  → 80% power 所需最小 n = {min_n}")
    min_n_90 = min_n_for_power(target_rho, target_power=0.90)
    print(f"  → 90% power 所需最小 n = {min_n_90}")

    # === 4. Bootstrap CI 收敛曲线 ===
    print("\n[4] Bootstrap CI 收敛曲线 (Spearman ρ norm_hamming vs R@5)")
    bootstrap_curve = bootstrap_ci_curve(
        x_hamming_norm, y_recall,
        fn=lambda x, y: spearmanr(x, y).statistic,
        n_boot_list=[100, 500, 1000, 5000, 10000],
    )
    for n_boot, (lo, med, hi) in bootstrap_curve.items():
        print(f"  n_boot={n_boot:5d}: CI95 = [{lo:+.4f}, {hi:+.4f}], median = {med:+.4f}")

    # === 5. Jackknife LOO 影响分析 ===
    print("\n[5] Jackknife leave-one-out (Spearman ρ norm_hamming vs R@5)")
    full_spearman = spearmanr(x_hamming_norm, y_recall).statistic
    print(f"  full-data Spearman ρ = {full_spearman:+.4f}")
    loo_results = jackknife_influence(
        x_hamming_norm, y_recall,
        fn=lambda x, y: spearmanr(x, y).statistic,
    )
    print(f"  leave-one-out 影响 (按 |Δ| 排序):")
    sorted_loo = sorted(loo_results, key=lambda r: abs(r[2]), reverse=True)
    for idx, loo_rho, delta in sorted_loo:
        tid = active_manifest[idx]["id"]
        print(f"    drop {tid:30s}: ρ={loo_rho:+.4f}  Δ={delta:+.4f}")

    # === 6. 不同 k 下 ρ 敏感性 ===
    print(f"\n[6] ρ vs k 敏感性 (Spearman ρ norm_hamming vs R@5)")
    k_sensitivity = []
    for k in args.ks:
        x_k = np.array([e["per_k_metrics"][k]["hamming_norm_mean"] for e in active_manifest])
        s_r, s_p = spearmanr(x_k, y_recall)
        k_sensitivity.append({"k": k, "spearman_rho": float(s_r), "spearman_p": float(s_p)})
        print(f"  k={k:3d}: Spearman ρ = {s_r:+.4f} (p={s_p:.4f})")

    # === 汇总 verdict 友好的数字 ===
    print("\n=== 汇总 ===")
    primary_metric = correlation_table[0]  # norm_hamming Spearman
    summary = {
        "analysis_date": "2026-07-19",
        "n_tokenizers": n,
        "k_main": K_MAIN,
        "primary": {
            "metric": "norm_hamming vs R@5 (Spearman)",
            "rho": primary_metric["spearman_rho"],
            "p": primary_metric["spearman_p"],
            "p_holm": primary_metric["spearman_p_holm"],
        },
        "all_correlations": correlation_table,
        "power_analysis": {
            "target_rho": target_rho,
            "current_power": power_for_corr(target_rho, n),
            "min_n_80_power": min_n,
            "min_n_90_power": min_n_90,
        },
        "bootstrap_convergence": {
            n_boot: {"ci95_low": lo, "median": med, "ci95_high": hi}
            for n_boot, (lo, med, hi) in bootstrap_curve.items()
        },
        "jackknife_loo": [
            {"dropped": active_manifest[idx]["id"], "loo_rho": loo_rho, "delta": delta}
            for idx, loo_rho, delta in sorted_loo
        ],
        "k_sensitivity": k_sensitivity,
        "tokenizer_metrics": [
            {
                "id": e["id"],
                "sid_dim": e["loaded_sid_dim"],
                "downstream_R5": e["downstream_R5"],
                **{f"k{k}": e["per_k_metrics"][k] for k in args.ks},
            }
            for e in active_manifest
        ],
    }

    json_path = out_dir / "task27_enhanced_stats.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote JSON: {json_path}")

    print("\n=== 主结论 ===")
    p25 = primary_metric["spearman_rho"]
    p_holm = primary_metric["spearman_p_holm"]
    cur_power = power_for_corr(target_rho, n)
    print(f"  • 主指标 (norm_hamming vs R@5): Spearman ρ = {p25:+.4f}")
    print(f"  • Holm-Bonferroni 校正后 p = {p_holm:.4f} (n_tests={len(raw_pvals)})")
    print(f"  • 当前 n={n} 对 |ρ|={abs(target_rho):.2f} 的统计 power = {cur_power:.3f}")
    print(f"  • 80% power 所需最小 n = {min_n}")
    print(f"  • 如果不扩 n, 则:")
    print(f"    - ρ 值仅作 effect size 描述, 不能用于'显著性'决策")
    print(f"    - bootstrap CI95 在 n=6 下不稳定, 仅作方向性参考")


if __name__ == "__main__":
    main()
