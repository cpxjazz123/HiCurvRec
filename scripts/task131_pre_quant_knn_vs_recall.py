"""Task #131 — phonism 4 seed 量化前 kNN vs Recall 相关性分析.

目标:
1. 计算 phonism Toys test set 上 sentence-t5-base dense retrieval (pre-quantization) R@5 / R@10
2. 与 4 seed 模型 Test R@5 / R@10 做对比 (model_R / pre_quant_R 比例)
3. Per-test-item: 在 embedding space 中真值 target 的 rank vs per-seed 模型 rank 的 Spearman 相关

设计:
- 复用 task27 的 helper 函数 (load_t5_embedding, compute_t5_topk_neighbors)
- 单文件可直接运行 (无 CLI 参数)
- 输出到 products/task131_pre_quant_knn_vs_recall/
"""

import os
import sys
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PHONISM_DATA = Path("/home/wlia0047/ar57/wenyu/genrec/dataset/amazon")
PHONISM_OUT = Path("/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys")
os.chdir(REPO_ROOT)


# =============================================================================
# Manifest: 4 seed 下游 Test R@5 / R@10 真值 (从 verdicts 抽取)
# =============================================================================
SEEDS = [
    {"seed": 42, "task": "#32", "R5": 0.03150, "R10": 0.04950, "best_epoch": 77, "ckpt": PHONISM_OUT / "seed42" / "best_model.pt"},
    {"seed": 123, "task": "#33", "R5": 0.03097, "R10": 0.04948, "best_epoch": 89, "ckpt": PHONISM_OUT / "seed123" / "best_model.pt"},
    {"seed": 7, "task": "#34", "R5": 0.02981, "R10": 0.04694, "best_epoch": 59, "ckpt": PHONISM_OUT / "seed7" / "best_model.pt"},
    {"seed": 2024, "task": "#35", "R5": 0.02822, "R10": 0.04766, "best_epoch": 62, "ckpt": PHONISM_OUT / "seed2024" / "best_model.pt"},
]


# =============================================================================
# 数据加载
# =============================================================================
def load_item_embeddings(parquet_path: Path) -> tuple[np.ndarray, list]:
    """Load sentence-t5-base item embeddings from parquet.

    Returns: (emb[N,768] float32, asin[N])
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    asins = df["ItemID"].tolist()
    emb = np.stack(df["embedding"].values).astype(np.float32)
    return emb, asins


def build_test_set():
    """Build phonism test set: list of (history_asins, target_asin) pairs.

    History = sorted train items per user (sequence based on unixReviewTime).
    Each test item becomes one (history, target) pair.

    Returns: list of dicts with keys: user, history_asins, target_asin, target_idx
    """
    print("加载 datamaps ...")
    with open(PHONISM_DATA / "raw/toys/datamaps.json", "r") as f:
        dm = json.load(f)
    # item2id maps asin -> str(int_id); parquet rows are 0-indexed
    # So row index = int(item2id[asin]) - 1
    asin_to_strid = dm["item2id"]

    print("加载 review_splits ...")
    with open(PHONISM_DATA / "raw/toys/review_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    # Build user -> train history (sorted by time)
    print("构建 user train history ...")
    user_history = defaultdict(list)
    for r in splits["train"]:
        if r["asin"] in asin_to_strid:
            user_history[r["reviewerID"]].append(r)
    for u in user_history:
        user_history[u].sort(key=lambda x: x["unixReviewTime"])

    print("构建 test samples ...")
    test_samples = []
    skipped_no_history = 0
    skipped_no_vocab = 0
    for r in splits["test"]:
        target_asin = r["asin"]
        user = r["reviewerID"]
        if target_asin not in asin_to_strid:
            skipped_no_vocab += 1
            continue
        if user not in user_history or len(user_history[user]) == 0:
            skipped_no_history += 1
            continue
        history_asins = [x["asin"] for x in user_history[user] if x["asin"] in asin_to_strid]
        # target row index in item_emb = strid - 1
        target_idx = int(asin_to_strid[target_asin]) - 1
        test_samples.append({
            "user": user,
            "history_asins": history_asins,
            "target_asin": target_asin,
            "target_idx": target_idx,
        })

    print(f"  total test reviews: {len(splits['test'])}")
    print(f"  skipped no vocab: {skipped_no_vocab}")
    print(f"  skipped no history: {skipped_no_history}")
    print(f"  usable test samples: {len(test_samples)}")

    # Also build user -> history embedding for pre-quantization kNN
    # We need item embedding indexed by asin_to_strid
    return test_samples, asin_to_strid


def compute_history_embeddings(test_samples, item_emb, asin_to_strid, emb_dim=768):
    """For each test sample, compute mean-pool of train history item embeddings.

    Returns: hist_emb (N, 768) tensor
    """
    n = len(test_samples)
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        # Mean-pool over history item embeddings (row idx = strid - 1)
        idxs = [int(asin_to_strid[a]) - 1 for a in s["history_asins"] if a in asin_to_strid]
        if not idxs:
            # No history: zero vector (rare)
            continue
        hist_emb[i] = item_emb[idxs].mean(axis=0)
    return hist_emb


def compute_pre_quant_recall(hist_emb, target_idxs, item_emb, k_list=(5, 10)):
    """For each test sample, compute pre-quantization R@K via dense retrieval.

    1. L2-normalize both hist_emb and item_emb
    2. Cosine similarity = hist_emb @ item_emb.T  (shape: N x N_items)
    3. For each test sample, get top-K indices, check if target_idx is in top-K
    4. Compute hit rate (= R@K)

    Returns: dict {K: hit_rate}, per_sample_hits {K: list of bool}
    """
    n = hist_emb.shape[0]
    n_items = item_emb.shape[0]

    # L2-normalize
    hist_norm = hist_emb / (np.linalg.norm(hist_emb, axis=1, keepdims=True) + 1e-12)
    item_norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)

    # Compute cosine similarity in batches to avoid OOM
    print(f"  Computing cosine sim ({n} x {n_items}) ...")
    batch_size = 256
    target_idxs_arr = np.asarray(target_idxs)
    results = {K: np.zeros(n, dtype=bool) for K in k_list}
    ranks = np.zeros(n, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = hist_norm[start:end]  # (B, 768)
        sim_chunk = chunk @ item_norm.T  # (B, N_items)
        max_k = max(k_list)
        # Get top-K indices (descending sim)
        top_idx = np.argpartition(-sim_chunk, max_k, axis=1)[:, :max_k]
        # Re-sort within top-K (each row independently)
        for row_i, abs_i in enumerate(range(start, end)):
            target = target_idxs_arr[abs_i]
            top_row = top_idx[row_i]  # (max_k,)
            # Sort within top-K by descending similarity
            sub_sim = sim_chunk[row_i, top_row]
            sorted_local = np.argsort(-sub_sim)
            top_row_sorted = top_row[sorted_local]
            # Full rank of target in entire corpus (descending similarity)
            full_rank = np.argsort(-sim_chunk[row_i])
            target_rank = np.where(full_rank == target)[0][0]
            ranks[abs_i] = target_rank
            for K in k_list:
                results[K][abs_i] = target in top_row_sorted[:K]

    hit_rates = {K: float(results[K].mean()) for K in k_list}
    return hit_rates, results, ranks


# =============================================================================
# Per-seed analysis
# =============================================================================
def per_seed_ratio(pre_quant_R, per_seed_R):
    """Compute model_R / pre_quant_R per seed.

    pre_quant_R is single number (constant across seeds).
    per_seed_R is array of 4 values.
    Returns: array of 4 ratios.
    """
    return per_seed_R / pre_quant_R


def rank_aggregation(ranks, group_ids):
    """Compute mean rank per group.

    Args:
        ranks: (N,) int array
        group_ids: (N,) int array, group IDs (e.g., per-user test items)

    Returns: dict {gid: mean_rank}
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for r, g in zip(ranks, group_ids):
        bucket[g].append(r)
    return {g: float(np.mean(v)) for g, v in bucket.items()}


# =============================================================================
# 主流程
# =============================================================================
def main():
    out_dir = Path("products/task131_pre_quant_knn_vs_recall")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = PHONISM_DATA / "processed/toys/item_emb_sentence-t5-base.parquet"

    print(f"=== Task #131: 量化前 kNN vs 4-seed Recall 相关性 ===")
    print(f"输出目录: {out_dir}")
    print()

    # Step 1: 加载 item embeddings
    print("[1] 加载 item embeddings ...")
    item_emb, asins = load_item_embeddings(parquet_path)
    print(f"  shape = {item_emb.shape}")
    assert item_emb.shape[0] == 11924
    assert item_emb.shape[1] == 768

    # Step 2: 构建 test set
    print("\n[2] 构建 test set ...")
    test_samples, asin_to_strid = build_test_set()
    target_idxs = [s["target_idx"] for s in test_samples]
    users = [s["user"] for s in test_samples]

    # Step 3: 计算 history embeddings
    print("\n[3] 计算 history embeddings (mean-pool) ...")
    hist_emb = compute_history_embeddings(test_samples, item_emb, asin_to_strid)
    print(f"  shape = {hist_emb.shape}")
    print(f"  norm mean = {np.linalg.norm(hist_emb, axis=1).mean():.4f}")

    # Step 4: 计算 pre-quantization test R@K
    print("\n[4] 计算 pre-quantization dense retrieval R@5 / R@10 ...")
    pre_quant_hits, per_sample_hits, pre_quant_ranks = compute_pre_quant_recall(
        hist_emb, target_idxs, item_emb, k_list=(5, 10, 50, 100)
    )
    print(f"  Pre-quant R@5  = {pre_quant_hits[5]:.6f}")
    print(f"  Pre-quant R@10 = {pre_quant_hits[10]:.6f}")
    print(f"  Pre-quant R@50 = {pre_quant_hits[50]:.6f}")
    print(f"  Pre-quant R@100= {pre_quant_hits[100]:.6f}")
    print(f"  Mean target rank in full corpus (lower = better) = {pre_quant_ranks.mean():.1f}")
    print(f"  Median rank = {np.median(pre_quant_ranks):.1f}")

    # Step 5: 4 seed 对比
    print("\n[5] 4 seed model R vs pre-quant R ...")
    print(f"  {'seed':<6} {'task':<8} {'Model R@5':<10} {'Model R@10':<11} {'Model/Pre@5':<13} {'Model/Pre@10':<13}")
    seed_R5 = np.array([s["R5"] for s in SEEDS])
    seed_R10 = np.array([s["R10"] for s in SEEDS])
    ratio_R5 = per_seed_ratio(pre_quant_hits[5], seed_R5)
    ratio_R10 = per_seed_ratio(pre_quant_hits[10], seed_R10)

    for s, r5, r10 in zip(SEEDS, ratio_R5, ratio_R10):
        print(
            f"  {s['seed']:<6} {s['task']:<8} "
            f"{s['R5']:<10.5f} {s['R10']:<11.5f} "
            f"{r5:<13.4f} {r10:<13.4f}"
        )
    print(f"  {'mean':<6} {'-':<8} "
          f"{seed_R5.mean():<10.5f} {seed_R10.mean():<11.5f} "
          f"{ratio_R5.mean():<13.4f} {ratio_R10.mean():<13.4f}")
    print(f"  {'std':<6} {'-':<8} "
          f"{seed_R5.std():<10.5f} {seed_R10.std():<11.5f} "
          f"{ratio_R5.std():<13.4f} {ratio_R10.std():<13.4f}")

    # Step 6: 与 pre-quant ratio 做 seed-level 相关性
    # pre_quant_ratio is constant across seeds (model_R / pre_quant_R); can't do corr with constant
    # Instead: per-test-item 分析 (cross-seed variance not used here)
    # Alternative: per-test-item, compute whether pre-quant hit correlates with seed R
    # Since R is at corpus level (single number), this becomes ratio per test item.
    #
    # Better: per-test-item, compute (target rank in embedding space) and see if items
    # with low rank (easy to retrieve) are also where models succeed.
    print("\n[6] Per-test-item: pre-quant rank vs target hit ...")
    print(f"  Per-test-item: pre-quant R@5 = {pre_quant_hits[5]:.6f}")
    print(f"  Per-test-item: avg pre-quant rank = {pre_quant_ranks.mean():.1f}")
    print(f"  Per-test-item: median pre-quant rank = {np.median(pre_quant_ranks):.1f}")

    # Bucket items by pre-quant rank quartile, check hit rate
    quartiles = np.percentile(pre_quant_ranks, [25, 50, 75])
    print(f"  Pre-quant rank quartiles: 25%={quartiles[0]:.0f}, 50%={quartiles[1]:.0f}, 75%={quartiles[2]:.0f}")
    bucket_labels = np.zeros(len(pre_quant_ranks), dtype=int)
    bucket_labels[pre_quant_ranks >= quartiles[0]] = 1
    bucket_labels[pre_quant_ranks >= quartiles[1]] = 2
    bucket_labels[pre_quant_ranks >= quartiles[2]] = 3
    print(f"  Pre-quant R@5 by rank quartile (lower rank = easier):")
    for q in range(4):
        mask = bucket_labels == q
        hit_rate = pre_sample_hit_5 = per_sample_hits[5][mask].mean() if mask.sum() > 0 else 0
        n = mask.sum()
        rank_lo = pre_quant_ranks[mask].min() if mask.sum() > 0 else 0
        rank_hi = pre_quant_ranks[mask].max() if mask.sum() > 0 else 0
        print(f"    Q{q} (n={n:5d}, rank {rank_lo:5d}-{rank_hi:5d}): pre-quant R@5 = {hit_rate:.4f}")

    # Step 7: Per-user test items with multi-test items: average rank per user
    print("\n[7] Per-user 平均 pre-quant rank (针对有多 test item 的用户) ...")
    user_id_map = {u: i for i, u in enumerate(users)}
    user_rank_mean = rank_aggregation(pre_quant_ranks, users)
    multi_test_users = {u: r for u, r in user_rank_mean.items() if sum(1 for v in users if v == u) > 1}
    print(f"  multi-test users (≥2 test items): {len(multi_test_users)}")
    if multi_test_users:
        multi_ranks = list(multi_test_users.values())
        print(f"  mean pre-quant rank (multi-test users): {np.mean(multi_ranks):.1f}")

    # Step 8: 写产物
    print("\n[8] 写产物 ...")
    summary = {
        "pre_quantization": {
            "embedding_source": str(parquet_path),
            "embedding_dim": int(item_emb.shape[1]),
            "n_items": int(item_emb.shape[0]),
            "n_test_samples": len(test_samples),
            "history_mean_pool": True,
            "pre_quant_R_at_5": pre_quant_hits[5],
            "pre_quant_R_at_10": pre_quant_hits[10],
            "pre_quant_R_at_50": pre_quant_hits[50],
            "pre_quant_R_at_100": pre_quant_hits[100],
            "mean_target_rank": float(pre_quant_ranks.mean()),
            "median_target_rank": float(np.median(pre_quant_ranks)),
            "rank_quartiles_25_50_75": [float(x) for x in quartiles],
        },
        "per_seed": [
            {
                "seed": s["seed"],
                "task": s["task"],
                "model_R_at_5": s["R5"],
                "model_R_at_10": s["R10"],
                "best_epoch": s["best_epoch"],
                "ratio_R5_over_pre_quant_R5": float(seed_R5[i] / pre_quant_hits[5]),
                "ratio_R10_over_pre_quant_R10": float(seed_R10[i] / pre_quant_hits[10]),
            }
            for i, s in enumerate(SEEDS)
        ],
        "aggregate": {
            "mean_model_R5": float(seed_R5.mean()),
            "std_model_R5": float(seed_R5.std()),
            "mean_model_R10": float(seed_R10.mean()),
            "std_model_R10": float(seed_R10.std()),
            "mean_ratio_R5": float(ratio_R5.mean()),
            "mean_ratio_R10": float(ratio_R10.mean()),
            "pre_quant_R5_minus_model_R5_gap": float(pre_quant_hits[5] - seed_R5.mean()),
            "pre_quant_R10_minus_model_R10_gap": float(pre_quant_hits[10] - seed_R10.mean()),
        },
    }
    summary_path = out_dir / "task131_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  写 summary: {summary_path}")

    # 保存 per-test-sample 数据供后续 per-item per-seed 分析
    per_sample_path = out_dir / "task131_per_test_sample.npz"
    np.savez(
        per_sample_path,
        target_idxs=np.asarray(target_idxs),
        users=np.asarray(users),
        pre_quant_rank=pre_quant_ranks,
        pre_quant_hit_5=per_sample_hits[5],
        pre_quant_hit_10=per_sample_hits[10],
        pre_quant_hit_50=per_sample_hits[50],
    )
    print(f"  写 per-test-sample data: {per_sample_path}")

    # Step 9: 画图
    print("\n[9] 画图 ...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Subplot 1: pre-quant R@K curve
    ax = axes[0]
    k_vals = [1, 5, 10, 20, 50, 100]
    pre_quant_curve = []
    for K in k_vals:
        # Re-compute or use stored
        if K in pre_quant_hits:
            pre_quant_curve.append(pre_quant_hits[K])
        else:
            # Re-compute
            hp, _, _ = compute_pre_quant_recall(hist_emb, target_idxs, item_emb, k_list=(K,))
            pre_quant_curve.append(hp[K])
    ax.plot(k_vals, pre_quant_curve, "o-", label="Pre-quant (sentence-t5-base dense)", linewidth=2)
    ax.axhline(y=SEEDS[0]["R5"], color="r", linestyle="--", label=f"Model R@5 (4-seed mean={seed_R5.mean():.4f})")
    ax.axhline(y=SEEDS[0]["R10"], color="g", linestyle="--", label=f"Model R@10 (4-seed mean={seed_R10.mean():.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("K (top-K)")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Pre-quant Dense Retrieval R@K\n(Toys test set, history mean-pool)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Per-seed ratio bar chart
    ax = axes[1]
    seeds_labels = [str(s["seed"]) for s in SEEDS]
    x = np.arange(len(seeds_labels))
    width = 0.35
    ax.bar(x - width/2, ratio_R5, width, label="Ratio R@5 (Model/Pre-quant)", color="steelblue")
    ax.bar(x + width/2, ratio_R10, width, label="Ratio R@10 (Model/Pre-quant)", color="darkorange")
    ax.axhline(y=ratio_R5.mean(), color="steelblue", linestyle="--", alpha=0.5, label=f"Mean ratio R@5 = {ratio_R5.mean():.4f}")
    ax.axhline(y=ratio_R10.mean(), color="darkorange", linestyle="--", alpha=0.5, label=f"Mean ratio R@10 = {ratio_R10.mean():.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds_labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Ratio = Model R / Pre-quant R")
    ax.set_title(f"Model / Pre-quant Ratio per Seed\n(Pre-quant R@5={pre_quant_hits[5]:.4f}, R@10={pre_quant_hits[10]:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Subplot 3: per-test-item pre-quant hit@5 distribution
    ax = axes[2]
    sorted_ranks = np.sort(pre_quant_ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    ax.plot(sorted_ranks, cdf, label="CDF of target rank")
    for K in [5, 10, 50, 100]:
        ax.axvline(x=K, color="gray", linestyle=":", alpha=0.5)
        ax.text(K, 0.95, f"R@{K}={pre_quant_hits[K]:.4f}", rotation=90, fontsize=8, va="top")
    ax.set_xlabel("Target item rank in dense retrieval (lower = better)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Per-test-item Pre-quant Rank Distribution\n(mean={pre_quant_ranks.mean():.1f}, median={np.median(pre_quant_ranks):.1f})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig_path = out_dir / "task131_pre_quant_vs_recall.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  写 figure: {fig_path}")

    print("\n=== Task #131 主分析完成 ===")
    print(f"  Pre-quant R@5  = {pre_quant_hits[5]:.6f}")
    print(f"  Pre-quant R@10 = {pre_quant_hits[10]:.6f}")
    print(f"  Model mean R@5 (4 seed)   = {seed_R5.mean():.6f}")
    print(f"  Model mean R@10 (4 seed)  = {seed_R10.mean():.6f}")
    print(f"  Ratio R@5 mean (model/pre-quant) = {ratio_R5.mean():.4f}")
    print(f"  Ratio R@10 mean (model/pre-quant) = {ratio_R10.mean():.4f}")
    print(f"\n  Summary: {summary_path}")
    print(f"  Figure:  {fig_path}")
    print(f"  Per-sample: {per_sample_path}")


if __name__ == "__main__":
    main()