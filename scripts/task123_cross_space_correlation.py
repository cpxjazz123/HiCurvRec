"""Task #123 — 跨空间 n=4 完整相关性分析.

按用户纠正 (2026-07-19):
- 跨空间比较**本身有效**, 只要每个 tokenizer 用自己训练空间测 kNN
- baseline (T5 输入) → T5 空间 kNN co-cluster
- m=0/m=1 (MCKG 输入) → MCKG 空间 kNN co-cluster
- 把所有 4 个 (kNN, R@5) 数据点合在一起, 算完整 n=4 Spearman ρ

启动:
    cd /home/wlia0047/ar57/wenyu/GeneRec
    python3 scripts/task123_cross_space_correlation.py
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO_ROOT)


# 4 个 tokenizer 跨空间数据 (合并 Task #119 T5 + Task #120 MCKG)
# 每个 tokenizer 用自己训练空间测 kNN co-cluster@10
DATA = [
    {
        "id": "Task85_m1_quasi_euclid",
        "label": "m=1 准欧氏 (κ≈-0.17)",
        "training_space": "MCKG",
        "kNN_space": "MCKG (κ-weighted)",
        "co_cluster_at_10": 0.3700,  # Task #120
        "downstream_R5": 0.0200,
        "downstream_R10": 0.0288,
    },
    {
        "id": "Task87_K256_seed42",
        "label": "baseline K=256 flat (TIGER-aligned)",
        "training_space": "T5",
        "kNN_space": "T5 (cosine)",
        "co_cluster_at_10": 0.9774,  # Task #119
        "downstream_R5": 0.01937,
        "downstream_R10": 0.03318,
    },
    {
        "id": "Task85_m0_sphere",
        "label": "m=0 球面 (κ≈+0.85)",
        "training_space": "MCKG",
        "kNN_space": "MCKG (κ-weighted)",
        "co_cluster_at_10": 0.2356,  # Task #120
        "downstream_R5": 0.0174,
        "downstream_R10": 0.0262,
    },
    {
        "id": "Task107_K256_seed123",
        "label": "baseline K=256 seed=123 (variant)",
        "training_space": "T5",
        "kNN_space": "T5 (cosine)",
        "co_cluster_at_10": 0.9774,  # Task #119 (同 baseline SID, 同 T5 kNN)
        "downstream_R5": 0.01489,
        "downstream_R10": 0.02627,
    },
]


def main():
    print("=" * 70)
    print("Task #123 — 跨空间 n=4 完整相关性分析")
    print("=" * 70)

    # 显示数据表
    print(f"\n{'ID':<28} {'TrainSpace':<6} {'kNN@10':<8} {'R@5':<8} {'R@10':<8}")
    print("-" * 70)
    for d in DATA:
        print(f"{d['id']:<28} {d['training_space']:<6} {d['co_cluster_at_10']:<8.4f} {d['downstream_R5']:<8.5f} {d['downstream_R10']:<8.5f}")

    # 提取 x (co-cluster@10) 和 y (R@5)
    x = np.array([d["co_cluster_at_10"] for d in DATA])
    y = np.array([d["downstream_R5"] for d in DATA])

    # 主相关性
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    kendall_t, kendall_p = kendalltau(x, y)

    print(f"\n=== 主分析 (n=4) ===")
    print(f"  x = co-cluster@10 in tokenizer 训练空间")
    print(f"  y = downstream R@5")
    print(f"  Pearson  ρ = {pearson_r:+.4f}  (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:+.4f}  (p={spearman_p:.4f})")
    print(f"  Kendall  τ = {kendall_t:+.4f}  (p={kendall_p:.4f})")

    # 同指标用 R@10
    y10 = np.array([d["downstream_R10"] for d in DATA])
    sp_r10, sp_p10 = spearmanr(x, y10)
    print(f"\n  [辅助] Spearman ρ (co-cluster@10 vs R@10) = {sp_r10:+.4f}  (p={sp_p10:.4f})")

    # 排序
    print(f"\n=== 排序对比 ===")
    sorted_by_cocluster = sorted(DATA, key=lambda d: -d["co_cluster_at_10"])
    sorted_by_r5 = sorted(DATA, key=lambda d: -d["downstream_R5"])
    print(f"  按 co-cluster@10 (高→低):")
    for i, d in enumerate(sorted_by_cocluster, 1):
        print(f"    {i}. {d['id']:<28} co-cluster={d['co_cluster_at_10']:.4f}")
    print(f"  按 R@5 (高→低):")
    for i, d in enumerate(sorted_by_r5, 1):
        print(f"    {i}. {d['id']:<28} R@5={d['downstream_R5']:.5f}")

    # 同输入空间子分析 (避免跨空间维数差混淆)
    print(f"\n=== 子分析: 同输入空间 ===")
    mckg_data = [d for d in DATA if d["training_space"] == "MCKG"]
    t5_data = [d for d in DATA if d["training_space"] == "T5"]
    print(f"  MCKG 空间 (n={len(mckg_data)}): m=1 vs m=0")
    print(f"    m=1: co-cluster={mckg_data[0]['co_cluster_at_10']:.4f}, R@5={mckg_data[0]['downstream_R5']:.5f}")
    print(f"    m=0: co-cluster={mckg_data[1]['co_cluster_at_10']:.4f}, R@5={mckg_data[1]['downstream_R5']:.5f}")
    print(f"    → 排序一致 (m=1 > m=0 在 co-cluster 和 R@5)")
    print(f"  T5 空间 (n={len(t5_data)}): baseline seed=42 vs seed=123")
    print(f"    seed=42: co-cluster={t5_data[0]['co_cluster_at_10']:.4f}, R@5={t5_data[0]['downstream_R5']:.5f}")
    print(f"    seed=123: co-cluster={t5_data[1]['co_cluster_at_10']:.4f}, R@5={t5_data[1]['downstream_R5']:.5f}")
    print(f"    → co-cluster 相同 (同 SID), R@5 不同 (seed variance)")

    # Caveats
    print(f"\n=== 注意事项 ===")
    print(f"  1. 跨空间比较本身有效, 但绝对值不能直接比:")
    print(f"     T5 768d 高维, co-cluster 偏大 (~0.98)")
    print(f"     MCKG 96d 低维, co-cluster 偏小 (~0.24-0.37)")
    print(f"  2. 但**排序**和**相关性**跨空间可比较 (单维标量)")
    print(f"  3. baseline seed=42 与 seed=123 共用 SID, co-cluster 相同, 唯一变量是 TIGER seed")
    print(f"  4. n=4 数据点, Spearman ρ p-value 反映样本量限制而非关系不存在")

    # 输出 JSON
    out = {
        "n": len(DATA),
        "primary": {
            "x_label": "co_cluster_at_10_in_tokenizer_training_space",
            "y_label": "downstream_R5",
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "kendall_tau": float(kendall_t),
            "kendall_p": float(kendall_p),
        },
        "secondary_R10": {
            "spearman_r": float(sp_r10),
            "spearman_p": float(sp_p10),
        },
        "data": DATA,
        "caveats": [
            "Cross-space absolute values incomparable (T5 768d high-dim → high co-cluster; MCKG 96d low-dim → lower co-cluster).",
            "Cross-space rank and correlation are still valid (1D scalar).",
            "n=4 → Spearman p-value reflects sample size, not relationship existence.",
            "Task #107 (seed=123) has identical co-cluster to Task #87 (same SID); R@5 difference is TIGER seed variance.",
        ],
        "interpretation": {
            "spearman_sign": "+" if spearman_r > 0 else "-",
            "ranking_consistent": sorted_by_cocluster[0]["id"] == sorted_by_r5[0]["id"] or
                                  (sorted_by_cocluster[0]["id"].startswith("Task85_m1") and sorted_by_r5[0]["id"].startswith("Task85_m1")),
        },
    }

    out_path = REPO_ROOT / "task123_cross_space_correlation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {out_path}")

    # CSV
    csv_path = REPO_ROOT / "task123_cross_space_table.csv"
    with open(csv_path, "w") as f:
        f.write("id,label,training_space,kNN_space,co_cluster_at_10,downstream_R5,downstream_R10\n")
        for d in DATA:
            f.write(f"{d['id']},{d['label']},{d['training_space']},{d['kNN_space']},{d['co_cluster_at_10']:.4f},{d['downstream_R5']:.5f},{d['downstream_R10']:.5f}\n")
    print(f"[save] {csv_path}")

    return out


if __name__ == "__main__":
    main()