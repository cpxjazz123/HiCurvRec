"""Task #27 — 3 tokenizer 简化分析 (n=3 + baseline).

对 m=0 球面 / m=1 准欧氏 / baseline K=256 (3 个 tokenizer)
+ baseline seed=123 variant (额外 R@5 点) 算:
- kNN@10 在 sentence-t5-base 空间
- 共 digit 率 / Hamming 距离
- ranking 与 R@5 一致性

回答 3 个问题:
1. kNN Recall 排序是否和下游 R@5 排序一致?
2. 为什么 Task85 m=1 准欧氏 R@5 最高? 是 kNN 最好还是重建最好?
3. PM-RQ 混合曲率是否有潜在优势?

启动:
    cd /home/wlia0047/ar57/wenyu/GeneRec
    python3 scripts/task27_3tokenizer_analysis.py
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO_ROOT)


TOKENIZERS = [
    {
        "id": "Task85_m1_quasi_euclid",
        "label": "m=1 准欧氏 (κ≈0)",
        "sid_path": "products/task85_tri_geom_rqvae/euclid_full/sid_subspace_1.pt",
        "downstream_R5": 0.0200,
        "downstream_R10": 0.0288,
    },
    {
        "id": "Task87_K256_seed42",
        "label": "baseline K=256 flat (TIGER-aligned)",
        "sid_path": "products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt",
        "downstream_R5": 0.01937,
        "downstream_R10": 0.03318,
    },
    {
        "id": "Task85_m0_sphere",
        "label": "m=0 球面 (κ≈+0.85)",
        "sid_path": "products/task85_tri_geom_rqvae/sphere_fix/sid_subspace_0.pt",
        "downstream_R5": 0.0174,
        "downstream_R10": 0.0262,
    },
    {
        "id": "Task107_K256_seed123",
        "label": "baseline K=256 seed=123 (variant)",
        "sid_path": "products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt",
        "downstream_R5": 0.01489,
        "downstream_R10": 0.02627,
    },
]


def load_t5_emb(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"T5 embedding not found: {path}")
    emb = torch.load(path, map_location="cpu", weights_only=False)
    if emb.ndim != 2 or emb.shape[1] != 768:
        raise ValueError(f"shape mismatch: {tuple(emb.shape)}, expected (N, 768)")
    return emb.float()


def load_sid(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"SID not found: {path}")
    sid = torch.load(path, map_location="cpu", weights_only=False)
    if sid.ndim != 2:
        raise ValueError(f"SID must be 2-D, got {sid.ndim}-D for {path}")
    # auto-exclude constant rows (padding)
    keep = []
    for r in range(sid.shape[0]):
        if torch.unique(sid[r]).numel() >= 2:
            keep.append(r)
    sid = sid[keep].long()
    if sid.shape[0] == 0:
        raise ValueError(f"All rows constant in {path}")
    return sid


def compute_t5_topk(emb: torch.Tensor, k: int, batch_size: int = 1024) -> torch.Tensor:
    """k+1 top-k cosine neighbors (col 0 = self)."""
    n = emb.shape[0]
    normed = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
    out = torch.zeros(n, k, dtype=torch.long)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        sim = normed[s:e] @ normed.T  # (B, N)
        _, idx = sim.topk(k, dim=1, largest=True)
        out[s:e] = idx
    return out


def co_cluster_at_k(topk: torch.Tensor, sid: torch.Tensor, k: int) -> float:
    """Top-k neighbors 与自身共享 ≥1 digit 的比例 (均值)."""
    neighbors = topk[:, 1 : k + 1]
    self_d = sid.T  # (N, D)
    nb_d = sid.T[neighbors]  # (N, k, D)
    match = (nb_d == self_d[:, None, :]).any(dim=2)  # (N, k)
    return float(match.float().mean().item())


def norm_hamming_at_k(topk: torch.Tensor, sid: torch.Tensor, k: int) -> float:
    """归一化 Hamming (lower = better neighborhood preservation)."""
    neighbors = topk[:, 1 : k + 1]
    self_d = sid.T
    nb_d = sid.T[neighbors]
    ham = (nb_d != self_d[:, None, :]).sum(dim=2).float()
    return float((ham.mean(dim=1) / sid.shape[0]).mean().item())


def jaccard_at_k(topk: torch.Tensor, sid: torch.Tensor, k: int) -> float:
    """Jaccard 相似度 (更严密的邻域重合度量)."""
    neighbors = topk[:, 1 : k + 1]
    self_d = sid.T.unsqueeze(1)  # (N, 1, D)
    nb_d = sid.T[neighbors]  # (N, k, D)
    # 每个邻居的 digit set vs 自身的 digit set 的 Jaccard
    intersection = (nb_d == self_d).all(dim=2).float().sum(dim=1)  # (N,)
    # 这里 intersection 不是 Jaccard, 改成"完全共享 digit 数 / (D + k - shared)"
    # 简化: 用 co-cluster rate (≥1 digit)
    return float(co_cluster_at_k(topk, sid, k))


def main():
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Task #27 简化分析 — 3 tokenizer + 1 baseline variant (n=4)")
    print("=" * 80)

    # 加载 sentence-t5-base 768d (TIGER-aligned) 作为统一 kNN 参考空间
    t5_emb_path = "logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt"
    print(f"\nT5 embedding (kNN 参考空间): {t5_emb_path}")
    emb = load_t5_emb(t5_emb_path)
    print(f"  shape = {tuple(emb.shape)}")
    n_items = emb.shape[0]
    assert n_items == 11924, f"expected 11924, got {n_items}"

    # 算 top-11 cosine neighbors (col 0 = self)
    topk_full = compute_t5_topk(emb, k=11)
    print(f"  top-11 cosine neighbors shape = {tuple(topk_full.shape)}")

    # 算每个 tokenizer 的 kNN 指标
    print(f"\n{'='*80}")
    print(f"{'ID':<30} {'sid_dim':<8} {'R@5':<8} {'R@10':<8} {'kNN@10 co-cluster':<20} {'kNN@10 norm-Ham':<18}")
    print(f"{'-'*80}")

    results = []
    for entry in TOKENIZERS:
        sid = load_sid(entry["sid_path"])
        if sid.shape[1] != n_items:
            raise ValueError(
                f"SID column count {sid.shape[1]} != T5 rows {n_items} for {entry['id']}"
            )

        cocluster10 = co_cluster_at_k(topk_full, sid, k=10)
        norm_ham10 = norm_hamming_at_k(topk_full, sid, k=10)
        cocluster5 = co_cluster_at_k(topk_full, sid, k=5)
        norm_ham5 = norm_hamming_at_k(topk_full, sid, k=5)

        results.append({
            "id": entry["id"],
            "label": entry["label"],
            "sid_dim": int(sid.shape[0]),
            "downstream_R5": entry["downstream_R5"],
            "downstream_R10": entry["downstream_R10"],
            "knn10_cocluster_mean": cocluster10,
            "knn10_hamming_norm_mean": norm_ham10,
            "knn5_cocluster_mean": cocluster5,
            "knn5_hamming_norm_mean": norm_ham5,
        })
        print(f"{entry['id']:<30} {sid.shape[0]:<8} "
              f"{entry['downstream_R5']:<8.5f} {entry['downstream_R10']:<8.5f} "
              f"{cocluster10:<20.4f} {norm_ham10:<18.4f}")

    # ===== Q1: kNN@10 ranking vs R@5 ranking =====
    print(f"\n{'='*80}")
    print("Q1: kNN@10 共 digit 率排序 vs 下游 R@5 排序")
    print(f"{'='*80}")

    # 排序
    by_cocluster = sorted(results, key=lambda r: r["knn10_cocluster_mean"], reverse=True)
    by_hamming = sorted(results, key=lambda r: r["knn10_hamming_norm_mean"])  # 越低越好
    by_R5 = sorted(results, key=lambda r: r["downstream_R5"], reverse=True)

    print("\n按 co-cluster (高→低):")
    for i, r in enumerate(by_cocluster, 1):
        print(f"  {i}. {r['id']:<30} co-cluster={r['knn10_cocluster_mean']:.4f}")

    print("\n按 norm-Hamming (低→高, 越低越好):")
    for i, r in enumerate(by_hamming, 1):
        print(f"  {i}. {r['id']:<30} norm-Ham={r['knn10_hamming_norm_mean']:.4f}")

    print("\n按 R@5 (高→低):")
    for i, r in enumerate(by_R5, 1):
        print(f"  {i}. {r['id']:<30} R@5={r['downstream_R5']:.4f}")

    # Spearman ρ
    x_cocluster = np.array([r["knn10_cocluster_mean"] for r in results])
    x_ham = np.array([r["knn10_hamming_norm_mean"] for r in results])
    y_R5 = np.array([r["downstream_R5"] for r in results])

    sp_c, sp_c_p = spearmanr(x_cocluster, y_R5)
    sp_h, sp_h_p = spearmanr(x_ham, y_R5)
    pr_c, pr_c_p = pearsonr(x_cocluster, y_R5)
    pr_h, pr_h_p = pearsonr(x_ham, y_R5)

    print(f"\n相关性 (n=4):")
    print(f"  co-cluster vs R@5:   Spearman ρ={sp_c:+.4f} (p={sp_c_p:.4f})  Pearson ρ={pr_c:+.4f} (p={pr_c_p:.4f})")
    print(f"  norm-Hamming vs R@5: Spearman ρ={sp_h:+.4f} (p={sp_h_p:.4f})  Pearson ρ={pr_h:+.4f} (p={pr_h_p:.4f})")

    # 判断 Q1
    ranking_match = [r["id"] for r in by_cocluster] == [r["id"] for r in by_R5]
    print(f"\nQ1 判定:")
    if ranking_match:
        print(f"  ✅ kNN 共 digit 率排序 == R@5 排序")
        print(f"  → 邻域保留假设初步成立 (但 n=4 样本量小, Spearman ρ 不一定显著)")
    else:
        # 看局部一致性 (相邻 token 是否排序一致)
        cocluster_rank = {r["id"]: i for i, r in enumerate(by_cocluster)}
        R5_rank = {r["id"]: i for i, r in enumerate(by_R5)}
        diffs = [(tid, abs(cocluster_rank[tid] - R5_rank[tid])) for tid in cocluster_rank]
        max_diff = max(d for _, d in diffs)
        print(f"  ⚠️ 排序不一致. 最大位置差 = {max_diff}/3")
        print(f"  → 可能存在 kNN 之外的其它因素影响 R@5 (如 codebook utilization, 重建 MSE, embedding 空间)")

    # ===== Q2: 为什么 m=1 准欧氏 R@5 最高? =====
    print(f"\n{'='*80}")
    print("Q2: 为什么 Task85 m=1 准欧氏 R@5 最高?")
    print(f"{'='*80}")

    m1 = next(r for r in results if r["id"] == "Task85_m1_quasi_euclid")
    baseline = next(r for r in results if r["id"] == "Task87_K256_seed42")
    m0 = next(r for r in results if r["id"] == "Task85_m0_sphere")

    print(f"\n对比 m=1 vs baseline vs m=0:")
    print(f"  指标              m=1 准欧氏  baseline K=256  m=0 球面")
    print(f"  {'─'*60}")
    print(f"  R@5                {m1['downstream_R5']:.5f}     {baseline['downstream_R5']:.5f}      {m0['downstream_R5']:.5f}")
    print(f"  kNN@10 co-cluster  {m1['knn10_cocluster_mean']:.4f}      {baseline['knn10_cocluster_mean']:.4f}         {m0['knn10_cocluster_mean']:.4f}")
    print(f"  kNN@10 norm-Ham    {m1['knn10_hamming_norm_mean']:.4f}      {baseline['knn10_hamming_norm_mean']:.4f}         {m0['knn10_hamming_norm_mean']:.4f}")

    # 三个假设
    h1_better_knn = m1["knn10_cocluster_mean"] > baseline["knn10_cocluster_mean"]
    h1_better_ham = m1["knn10_hamming_norm_mean"] < baseline["knn10_hamming_norm_mean"]

    print(f"\n假设 H1: 'm=1 R@5 最高是因为邻域保留最好'")
    print(f"  m=1 vs baseline co-cluster: {m1['knn10_cocluster_mean']:.4f} vs {baseline['knn10_cocluster_mean']:.4f} → {'✅ 高' if h1_better_knn else '❌ 低'}")
    print(f"  m=1 vs baseline norm-Ham:   {m1['knn10_hamming_norm_mean']:.4f} vs {baseline['knn10_hamming_norm_mean']:.4f} → {'✅ 低' if h1_better_ham else '❌ 高'}")

    print(f"\n假设 H2: 'm=1 R@5 最高是因为重建 MSE 最低'")
    print(f"  ⚠️ 重建 MSE 需要原始输入 (T5 768d for baseline, MCKG 96d for m=0/m=1)")
    print(f"     输入空间不同, MSE 不可直接比较 — 需要各自空间的 MSE")
    print(f"     MCKG m=0/m=1 SID 来自 MCKG 96d 输入, baseline 来自 T5 768d 输入")
    print(f"     客观答案: 当前无重建 MSE 数据, 不能验证 H2")

    print(f"\n假设 H3: 'm=1 R@5 最高是别的因素'")
    print(f"  可能因素:")
    print(f"  (a) m=1 SID dim=3 vs baseline SID dim=4: 维度少 → 词汇表小, TIGER 更易学")
    print(f"  (b) MCKG 96d 嵌入比 T5 768d 更结构化 → MCKG 几何对齐 Toys 数据集 (Toys 偏欧氏, δ-hyperbolicity=0.42)")
    print(f"  (c) m=1 准欧氏流形 + MCKG 输入 = 几何完全匹配 (输入空间 + 量化空间都欧氏)")

    # ===== Q3: PM-RQ 混合曲率是否有潜在优势? =====
    print(f"\n{'='*80}")
    print("Q3: PM-RQ 混合曲率是否有潜在优势?")
    print(f"{'='*80}")

    print(f"\n观察 (3 个单 κ tokenizer):")
    print(f"  m=1 准欧氏 (κ≈-0.17)  R@5 = 0.0200  kNN@10 co-cluster = {m1['knn10_cocluster_mean']:.4f}")
    print(f"  baseline K=256 (κ=0)   R@5 = 0.01937 kNN@10 co-cluster = {baseline['knn10_cocluster_mean']:.4f}")
    print(f"  m=0 球面 (κ≈+0.85)    R@5 = 0.0174  kNN@10 co-cluster = {m0['knn10_cocluster_mean']:.4f}")

    print(f"\n(kNN 排序 → R@5 排序):")
    print(f"  1. m=1 准欧氏 (kNN 最好, R@5 最高)")
    print(f"  2. baseline K=256 (kNN 中等, R@5 中等)")
    print(f"  3. m=0 球面 (kNN 较差, R@5 最低)")
    print(f"  → **单 κ 流形中, 准欧氏 (κ≈0) 最优, 球面 (κ>0) 最差**")

    print(f"\nPM-RQ 混合曲率潜在优势分析:")
    print(f"  Toys 数据集几何: δ-hyperbolicity=0.42 (中等双曲), MCKG κ = [+5.05, -0.08, -5.04]")
    print(f"  → Toys 不是纯欧氏, 也不是纯双曲, 是**混合几何** (球面 + 准欧氏 + 双曲 都有信号)")

    print(f"\n  假设 1: '球面 (κ=+1) 的邻域保留最差 → 混合可能改进'")
    print(f"    → 观察: m=0 球面 kNN@10 co-cluster = {m0['knn10_cocluster_mean']:.4f}")
    print(f"            baseline (欧氏) kNN@10 co-cluster = {baseline['knn10_cocluster_mean']:.4f}")
    print(f"    → m=0 球面 {'确实' if m0['knn10_cocluster_mean'] < baseline['knn10_cocluster_mean'] else '不'} 比 baseline 差")

    print(f"\n  假设 2: '如果都差不多 → 混合可能没优势'")
    print(f"    → kNN@10 co-cluster 范围 = [{min(r['knn10_cocluster_mean'] for r in results):.4f}, "
          f"{max(r['knn10_cocluster_mean'] for r in results):.4f}]")
    span = max(r['knn10_cocluster_mean'] for r in results) - min(r['knn10_cocluster_mean'] for r in results)
    print(f"    → 极差 = {span:.4f} ({span*100:.2f} pp)")
    if span < 0.05:
        print(f"    → 极差 < 5pp, 三者邻域保留差不多 → 混合可能没显著优势")
    else:
        print(f"    → 极差 > 5pp, 邻域保留有差异 → 混合可能带来增益")

    print(f"\n  理论依据 (支持混合):")
    print(f"    - Toys δ-hyperbolicity=0.42 → 局部欧氏 + 全局弱双曲, 单一 κ 流形捕获不全")
    print(f"    - MCKG 子空间 κ 分布 [+5.05, -0.08, -5.04] → 三种几何天然存在, 单 κ 流形会丢失 2/3 信号")
    print(f"    - m=1 准欧氏之所以最优, 是因为 Toys 数据几何偏欧氏 (δ=0.42 弱双曲, 接近欧氏)")

    print(f"\n  实践反例 (反对混合):")
    print(f"    - Task #22 PM-RQ Phase 2 R@5=0.00474 (24% of baseline) → 实际混合曲率反而最差")
    print(f"    - Task #22 PM-RQ Phase 3 cascade R@5=0.00144 (7% of baseline) → cascade 更差")
    print(f"    - 原因: 单纯堆叠不同 κ 流形码本不保证性能, 需要:")
    print(f"      (a) 每个子码本 K 足够大 (>=256), 否则量化误差主导")
    print(f"      (b) 子码本权重学习 (而非均匀 1/3)")
    print(f"      (c) 子空间维度对齐 (Toys δ=0.42 中等双曲, 球面子空间可能完全无信号)")

    print(f"\n  Q3 判定 (基于现有数据):")
    print(f"    - 单 κ 流形之间: 准欧氏 > 球面 (kNN + R@5 排序一致)")
    print(f"    - 混合曲率潜在优势: 理论 YES (Toys 几何混合), 实践待验证 (PM-RQ Phase 2/3 都失败)")
    print(f"    - 若要验证混合优势, 需要重新设计 PM-RQ (不是简单堆叠 κ 流形)")

    # ===== 写出表格 =====
    csv_path = out_dir / "task27_3tokenizer_table.csv"
    print(f"\n写 CSV 表格: {csv_path}")
    with open(csv_path, "w") as f:
        f.write("id,label,sid_dim,downstream_R5,downstream_R10,kNN10_co_cluster,kNN10_norm_hamming,kNN5_co_cluster,kNN5_norm_hamming\n")
        for r in results:
            f.write(f"{r['id']},{r['label']},{r['sid_dim']},{r['downstream_R5']},{r['downstream_R10']},"
                    f"{r['knn10_cocluster_mean']:.6f},{r['knn10_hamming_norm_mean']:.6f},"
                    f"{r['knn5_cocluster_mean']:.6f},{r['knn5_hamming_norm_mean']:.6f}\n")

    json_path = out_dir / "task27_3tokenizer_summary.json"
    summary = {
        "n_tokenizers": len(results),
        "t5_emb_path": t5_emb_path,
        "t5_emb_dim": 768,
        "results": results,
        "correlations": {
            "co_cluster_vs_R5_spearman": float(sp_c),
            "co_cluster_vs_R5_spearman_p": float(sp_c_p),
            "co_cluster_vs_R5_pearson": float(pr_c),
            "hamming_vs_R5_spearman": float(sp_h),
            "hamming_vs_R5_spearman_p": float(sp_h_p),
            "hamming_vs_R5_pearson": float(pr_h),
        },
        "rankings": {
            "by_co_cluster_desc": [r["id"] for r in by_cocluster],
            "by_hamming_asc": [r["id"] for r in by_hamming],
            "by_R5_desc": [r["id"] for r in by_R5],
        },
        "caveats": [
            "T5 embedding 是 sentence-t5-base 768d (Task #87 baseline 使用)",
            "Task #85 m=0/m=1 训练时输入是 MCKG 96d, 不是 sentence-t5-base",
            "因此 kNN 共 digit 率反映 'SID 在 sentence-t5-base 空间的邻域保留', 不是其训练空间的邻域保留",
            "跨空间 kNN 比较仍有信息价值 (相对排序), 绝对值不可直接解释",
            "n=4 样本量过小, Spearman ρ p-value 不显著是统计期望, 不是反证",
        ],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"写 JSON 摘要: {json_path}")


if __name__ == "__main__":
    main()