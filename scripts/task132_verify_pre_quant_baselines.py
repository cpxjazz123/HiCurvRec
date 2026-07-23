"""Task #132 — 验证 Task #131 pre-quant R@5=0.00107 是否合理.

对照方法:
1. 多种 history 聚合方式 (last-item, last-3 mean, recency-weighted, max-pool)
2. Random baseline (sanity check)
3. Test item leakage check
4. Cosine similarity 分布分析
5. Item-item kNN baseline (Task #27 style)
6. 按 popularity 分桶的 pre-quant R@K
"""

import os
import sys
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PHONISM_DATA = Path("/home/wlia0047/ar57/wenyu/genrec/dataset/amazon")
PHONISM_OUT = Path("/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys")
os.chdir(REPO_ROOT)


# =============================================================================
# 数据加载
# =============================================================================
def load_item_embeddings(parquet_path: Path) -> tuple[np.ndarray, list]:
    df = pd.read_parquet(parquet_path)
    asins = df["ItemID"].tolist()
    emb = np.stack(df["embedding"].values).astype(np.float32)
    return emb, asins


def build_test_set():
    """Build test set with sorted user history + target item."""
    with open(PHONISM_DATA / "raw/toys/datamaps.json", "r") as f:
        dm = json.load(f)
    asin_to_strid = dm["item2id"]
    id2item = dm["id2item"]

    with open(PHONISM_DATA / "raw/toys/review_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    user_history = defaultdict(list)
    for r in splits["train"]:
        if r["asin"] in asin_to_strid:
            user_history[r["reviewerID"]].append(r)
    for u in user_history:
        user_history[u].sort(key=lambda x: x["unixReviewTime"])

    test_samples = []
    for r in splits["test"]:
        target_asin = r["asin"]
        user = r["reviewerID"]
        if target_asin not in asin_to_strid:
            continue
        if user not in user_history or len(user_history[user]) == 0:
            continue
        history = sorted(user_history[user], key=lambda x: x["unixReviewTime"])
        history_asins = [x["asin"] for x in history if x["asin"] in asin_to_strid]
        target_idx = int(asin_to_strid[target_asin]) - 1
        test_samples.append({
            "user": user,
            "history_asins": history_asins,
            "history_lens": len(history_asins),
            "target_asin": target_asin,
            "target_idx": target_idx,
        })

    return test_samples, asin_to_strid, splits


# =============================================================================
# Aggregation methods
# =============================================================================
def agg_last_item(test_samples, item_emb, asin_to_strid):
    """Last-item only: use the most recent item embedding as query."""
    n = len(test_samples)
    emb_dim = item_emb.shape[1]
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        if s["history_asins"]:
            last_asin = s["history_asins"][-1]
            idx = int(asin_to_strid[last_asin]) - 1
            hist_emb[i] = item_emb[idx]
    return hist_emb


def agg_last_k_mean(test_samples, item_emb, asin_to_strid, k=3):
    """Last-K items mean."""
    n = len(test_samples)
    emb_dim = item_emb.shape[1]
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        if s["history_asins"]:
            recent = s["history_asins"][-k:]
            idxs = [int(asin_to_strid[a]) - 1 for a in recent]
            hist_emb[i] = item_emb[idxs].mean(axis=0)
    return hist_emb


def agg_mean_pool(test_samples, item_emb, asin_to_strid):
    """All-history mean (Task #131 default)."""
    n = len(test_samples)
    emb_dim = item_emb.shape[1]
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        if s["history_asins"]:
            idxs = [int(asin_to_strid[a]) - 1 for a in s["history_asins"]]
            hist_emb[i] = item_emb[idxs].mean(axis=0)
    return hist_emb


def agg_recency_weighted(test_samples, item_emb, asin_to_strid, alpha=0.7):
    """Recency-weighted: w_i = alpha^(N-1-i) where i is time position, N=history length."""
    n = len(test_samples)
    emb_dim = item_emb.shape[1]
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        if s["history_asins"]:
            N = len(s["history_asins"])
            weights = np.array([alpha ** (N - 1 - j) for j in range(N)], dtype=np.float32)
            weights /= weights.sum()
            idxs = [int(asin_to_strid[a]) - 1 for a in s["history_asins"]]
            hist_emb[i] = (weights[:, None] * item_emb[idxs]).sum(axis=0)
    return hist_emb


def agg_max_pool(test_samples, item_emb, asin_to_strid):
    """Max-pool over history (per-dim max)."""
    n = len(test_samples)
    emb_dim = item_emb.shape[1]
    hist_emb = np.zeros((n, emb_dim), dtype=np.float32)
    for i, s in enumerate(test_samples):
        if s["history_asins"]:
            idxs = [int(asin_to_strid[a]) - 1 for a in s["history_asins"]]
            hist_emb[i] = item_emb[idxs].max(axis=0)
    return hist_emb


def agg_random_query(item_emb, n_queries, seed=42):
    """Random item embedding as query (sanity baseline)."""
    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(item_emb.shape[0], size=n_queries, replace=True)
    return item_emb[rand_idx].copy()


# =============================================================================
# Retrieval evaluation
# =============================================================================
def evaluate_retrieval(query_emb, target_idxs, item_emb, k_list=(5, 10, 50, 100)):
    """Compute R@K + per-test-sample rank for a given query embedding."""
    n = query_emb.shape[0]
    n_items = item_emb.shape[0]
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    item_norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)

    target_idxs_arr = np.asarray(target_idxs)
    results = {K: np.zeros(n, dtype=bool) for K in k_list}
    ranks = np.zeros(n, dtype=np.int64)

    batch_size = 256
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = query_norm[start:end]
        sim_chunk = chunk @ item_norm.T
        max_k = max(k_list)
        top_idx = np.argpartition(-sim_chunk, max_k, axis=1)[:, :max_k]
        for row_i, abs_i in enumerate(range(start, end)):
            target = target_idxs_arr[abs_i]
            top_row = top_idx[row_i]
            sub_sim = sim_chunk[row_i, top_row]
            sorted_local = np.argsort(-sub_sim)
            top_row_sorted = top_row[sorted_local]
            full_rank = np.argsort(-sim_chunk[row_i])
            ranks[abs_i] = np.where(full_rank == target)[0][0]
            for K in k_list:
                results[K][abs_i] = target in top_row_sorted[:K]

    hit_rates = {K: float(results[K].mean()) for K in k_list}
    return hit_rates, results, ranks


def compute_true_random_R_at_K(n_items, k_list, n_trials=10000, seed=42):
    """Compute expected R@K for random query (truly uniform random retrieval)."""
    rng = np.random.default_rng(seed)
    results = {K: 0 for K in k_list}
    for _ in range(n_trials):
        target = rng.integers(0, n_items)
        # Random rank: uniformly in [0, n_items-1]
        rank = rng.integers(0, n_items)
        for K in k_list:
            if rank < K:
                results[K] += 1
    return {K: results[K] / n_trials for K in k_list}


# =============================================================================
# Sanity checks
# =============================================================================
def compute_item_popularity(splits, asin_to_strid):
    """Count train occurrences per item."""
    counter = Counter()
    for r in splits["train"]:
        if r["asin"] in asin_to_strid:
            counter[r["asin"]] += 1
    return counter


def check_test_leakage(test_samples, train_pop):
    """Check if test items appeared in train (should be normal, not leakage)."""
    in_train = sum(1 for s in test_samples if s["target_asin"] in train_pop)
    return in_train, len(test_samples)


def main():
    out_dir = Path("products/task132_verify_pre_quant")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = PHONISM_DATA / "processed/toys/item_emb_sentence-t5-base.parquet"

    print("=== Task #132: 验证 Task #131 pre-quant R@5=0.00107 是否合理 ===\n")

    print("[1] 加载 item embeddings ...")
    item_emb, asins = load_item_embeddings(parquet_path)
    print(f"  shape = {item_emb.shape}")

    print("\n[2] 构建 test set ...")
    test_samples, asin_to_strid, splits = build_test_set()
    target_idxs = [s["target_idx"] for s in test_samples]
    n_test = len(test_samples)
    n_items = item_emb.shape[0]
    print(f"  test samples = {n_test}, items = {n_items}")

    print("\n[3] Sanity checks ...")
    # True random baseline
    random_R = compute_true_random_R_at_K(n_items, k_list=(5, 10, 50, 100), n_trials=20000)
    print(f"  True random R@K (n=20000 trials):")
    for K, r in random_R.items():
        print(f"    R@{K} = {r:.6f}  (= {K}/{n_items} = {K/n_items:.6f})")

    # Leakage check
    train_pop = compute_item_popularity(splits, asin_to_strid)
    in_train, total = check_test_leakage(test_samples, train_pop)
    print(f"  Test items that appeared in train: {in_train}/{total} ({in_train/total*100:.1f}%)")
    print(f"  (正常情况: 100%, 因为 user-history 包含该 user 过去买的 items; test 是新 review)")

    # Test sample history length distribution
    hist_lens = [s["history_lens"] for s in test_samples]
    print(f"  History length: mean={np.mean(hist_lens):.2f}, median={np.median(hist_lens):.1f}, "
          f"min={min(hist_lens)}, max={max(hist_lens)}")

    # Last-item vs mean-pool vs random sanity: are they L2-normalized?
    print("\n[4] Item embedding norm check ...")
    item_norms = np.linalg.norm(item_emb, axis=1)
    print(f"  Item embedding L2 norm: mean={item_norms.mean():.4f}, std={item_norms.std():.4f}")
    print(f"  (sentence-t5-base 输出未 L2-normalize, norm ≈ 1)")

    print("\n[5] 多 history aggregation R@K 对比 ...")
    aggregations = {
        "last_item_only": agg_last_item(test_samples, item_emb, asin_to_strid),
        "last_3_mean": agg_last_k_mean(test_samples, item_emb, asin_to_strid, k=3),
        "last_5_mean": agg_last_k_mean(test_samples, item_emb, asin_to_strid, k=5),
        "mean_pool_all": agg_mean_pool(test_samples, item_emb, asin_to_strid),
        "recency_weighted_alpha07": agg_recency_weighted(test_samples, item_emb, asin_to_strid, alpha=0.7),
        "recency_weighted_alpha03": agg_recency_weighted(test_samples, item_emb, asin_to_strid, alpha=0.3),
        "max_pool": agg_max_pool(test_samples, item_emb, asin_to_strid),
    }

    # Add random baseline (using actual item embeddings as random queries)
    random_query_emb = agg_random_query(item_emb, n_test, seed=42)
    aggregations["random_query"] = random_query_emb

    summary = {}
    for name, query_emb in aggregations.items():
        hit_rates, per_sample_hits, ranks = evaluate_retrieval(
            query_emb, target_idxs, item_emb, k_list=(5, 10, 50, 100)
        )
        # Mean rank over the corpus (random = n_items/2 = 5962)
        mean_rank = float(ranks.mean())
        median_rank = float(np.median(ranks))
        summary[name] = {
            "R@5": hit_rates[5],
            "R@10": hit_rates[10],
            "R@50": hit_rates[50],
            "R@100": hit_rates[100],
            "mean_rank": mean_rank,
            "median_rank": median_rank,
        }
        print(f"\n  [{name}]")
        print(f"    R@5={hit_rates[5]:.6f}, R@10={hit_rates[10]:.6f}, R@50={hit_rates[50]:.6f}, R@100={hit_rates[100]:.6f}")
        print(f"    mean rank={mean_rank:.1f}, median rank={median_rank:.1f}")

    summary["true_random_baseline"] = {
        "R@5": random_R[5],
        "R@10": random_R[10],
        "R@50": random_R[50],
        "R@100": random_R[100],
        "mean_rank": n_items / 2.0,
        "median_rank": n_items / 2.0,
    }

    print("\n[6] Cosine similarity 分布 (mean-pool query vs random/target) ...")
    # Sample 1000 test samples
    sample_idx = np.random.default_rng(42).choice(n_test, size=min(1000, n_test), replace=False)
    mp_query = aggregations["mean_pool_all"][sample_idx]
    target_emb = item_emb[np.asarray(target_idxs)[sample_idx]]

    mp_norm = mp_query / (np.linalg.norm(mp_query, axis=1, keepdims=True) + 1e-12)
    target_norm = target_emb / (np.linalg.norm(target_emb, axis=1, keepdims=True) + 1e-12)
    item_norm = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)

    sim_to_target = (mp_norm * target_norm).sum(axis=1)
    # Random sim: each query vs a random item
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(n_items, size=1000)
    sim_to_random = (mp_norm * item_norm[rand_idx]).sum(axis=1)
    # Top-1 sim: each query vs its closest item
    sim_full = mp_norm @ item_norm.T  # (1000, N_items)
    sim_to_top1 = sim_full.max(axis=1)
    sim_top10 = np.partition(-sim_full, 10, axis=1)[:, :10]
    sim_to_top10_mean = -sim_top10.mean(axis=1)

    print(f"  Cosine sim: mean-pool query vs target  : mean={sim_to_target.mean():.4f}, std={sim_to_target.std():.4f}")
    print(f"  Cosine sim: mean-pool query vs random  : mean={sim_to_random.mean():.4f}, std={sim_to_random.std():.4f}")
    print(f"  Cosine sim: mean-pool query vs top-1   : mean={sim_to_top1.mean():.4f}")
    print(f"  Cosine sim: mean-pool query vs top-10 mean: {sim_to_top10_mean.mean():.4f}")
    print(f"  → 如果 target sim 不显著高于 random, dense retrieval 无判别力")

    print("\n[7] 按 item popularity 分桶的 pre-quant R@5 ...")
    pop_counter = train_pop
    # Define buckets
    test_targets_pop = np.array([pop_counter.get(s["target_asin"], 0) for s in test_samples])
    # Quantile-based buckets
    pop_quartiles = np.percentile(test_targets_pop, [25, 50, 75])
    print(f"  Test target popularity quartiles: 25%={pop_quartiles[0]:.0f}, 50%={pop_quartiles[1]:.0f}, 75%={pop_quartiles[2]:.0f}")
    pop_buckets = np.digitize(test_targets_pop, pop_quartiles)

    # Use last-item aggregation (best signal)
    last_query = aggregations["last_item_only"]
    last_norm = last_query / (np.linalg.norm(last_query, axis=1, keepdims=True) + 1e-12)
    item_norm_full = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12)

    pop_R5 = {}
    for q in range(5):
        mask = pop_buckets == q
        n_in = mask.sum()
        if n_in == 0:
            continue
        # Compute R@5 for this bucket
        sub_query = last_norm[mask]
        sub_target = np.asarray(target_idxs)[mask]
        sim = sub_query @ item_norm_full.T
        top5 = np.argpartition(-sim, 5, axis=1)[:, :5]
        hits = np.array([t in top5[i] for i, t in enumerate(sub_target)])
        pop_R5[f"Q{q}_n={n_in}_pop[{test_targets_pop[mask].min()}-{test_targets_pop[mask].max()}]"] = float(hits.mean())
        print(f"    Q{q} (n={n_in}, popularity [{test_targets_pop[mask].min()}-{test_targets_pop[mask].max()}]): last-item R@5 = {hits.mean():.4f}")

    # Item-item kNN baseline: for each test target, find top-K neighbors; check if history contains any of them
    print("\n[8] Item-item kNN baseline (Task #27 style) ...")
    # Compute top-K neighbors for each item in corpus
    K = 50
    sim_all = item_norm @ item_norm.T
    np.fill_diagonal(sim_all, -np.inf)
    item_topk = np.argpartition(-sim_all, K, axis=1)[:, :K]
    print(f"  Item-item top-{K} computed (N={n_items})")

    # For each test sample, check if any history item is in target's top-K neighbors
    iiknn_hits_5 = np.zeros(n_test, dtype=bool)
    iiknn_hits_10 = np.zeros(n_test, dtype=bool)
    for i, s in enumerate(test_samples):
        target = s["target_idx"]
        target_neighbors = item_topk[target]
        hist_idxs = set([int(asin_to_strid[a]) - 1 for a in s["history_asins"]])
        # Is any history item in target's top-K?
        overlap = hist_idxs.intersection(set(target_neighbors[:50]))
        # This is "history items are similar to target" metric, not retrieval
        # Better: if target's top-1 neighbor is in history
        if target_neighbors[0] in hist_idxs:
            iiknn_hits_5[i] = True
        if any(n in hist_idxs for n in target_neighbors[:10]):
            iiknn_hits_10[i] = True
    print(f"  History contains target's top-1 neighbor: {iiknn_hits_5.mean():.4f}")
    print(f"  History contains target's top-10 neighbor: {iiknn_hits_10.mean():.4f}")

    print("\n[9] 写产物 ...")
    # Save summary
    with open(out_dir / "baselines_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    import csv
    with open(out_dir / "baselines_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "R@5", "R@10", "R@50", "R@100", "mean_rank", "median_rank"])
        for name, m in summary.items():
            writer.writerow([
                name,
                f"{m['R@5']:.6f}",
                f"{m['R@10']:.6f}",
                f"{m['R@50']:.6f}",
                f"{m['R@100']:.6f}",
                f"{m['mean_rank']:.2f}",
                f"{m['median_rank']:.2f}",
            ])

    # Sanity check JSON
    sanity = {
        "true_random_R_at_K": random_R,
        "test_leakage": {"in_train": in_train, "total": total, "pct": in_train/total*100},
        "history_length": {
            "mean": float(np.mean(hist_lens)),
            "median": float(np.median(hist_lens)),
            "min": int(min(hist_lens)),
            "max": int(max(hist_lens)),
        },
        "item_emb_norm": {
            "mean": float(item_norms.mean()),
            "std": float(item_norms.std()),
        },
        "cosine_sim_distribution": {
            "query_vs_target": {
                "mean": float(sim_to_target.mean()),
                "std": float(sim_to_target.std()),
            },
            "query_vs_random": {
                "mean": float(sim_to_random.mean()),
                "std": float(sim_to_random.std()),
            },
            "query_vs_top1": {
                "mean": float(sim_to_top1.mean()),
            },
            "query_vs_top10_mean": {
                "mean": float(sim_to_top10_mean.mean()),
            },
        },
        "popularity_buckets_last_item_R5": pop_R5,
        "item_item_knn": {
            "history_contains_top1_neighbor": float(iiknn_hits_5.mean()),
            "history_contains_top10_neighbor": float(iiknn_hits_10.mean()),
        },
    }
    with open(out_dir / "sanity_checks.json", "w") as f:
        json.dump(sanity, f, indent=2)

    # Plot: bar chart of R@5 across methods
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    method_names = list(summary.keys())
    R5_vals = [summary[m]["R@5"] for m in method_names]
    R10_vals = [summary[m]["R@10"] for m in method_names]
    x = np.arange(len(method_names))
    width = 0.35
    ax.bar(x - width/2, R5_vals, width, label="R@5", color="steelblue")
    ax.bar(x + width/2, R10_vals, width, label="R@10", color="darkorange")
    ax.axhline(y=5/n_items, color="gray", linestyle="--", alpha=0.5, label=f"random baseline = 5/{n_items} = {5/n_items:.6f}")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R@K")
    ax.set_yscale("log")
    ax.set_title("Pre-quant R@K by aggregation method (Toys test)\n[log scale; mean-pool = Task #131 finding]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Cosine sim distribution
    ax = axes[1]
    ax.hist(sim_to_target, bins=50, alpha=0.5, label="mean-pool query → target", color="red")
    ax.hist(sim_to_random, bins=50, alpha=0.5, label="mean-pool query → random item", color="blue")
    ax.hist(sim_to_top1, bins=50, alpha=0.5, label="mean-pool query → top-1 item", color="green")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Cosine similarity distribution (1000 sample test queries)\n[target and random should differ for dense retrieval to work]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "task132_baselines_comparison.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  写 summary: {out_dir / 'baselines_comparison.json'}")
    print(f"  写 CSV: {out_dir / 'baselines_comparison.csv'}")
    print(f"  写 sanity: {out_dir / 'sanity_checks.json'}")
    print(f"  写 figure: {fig_path}")
    print("\n=== Task #132 完成 ===")

    # Print final verdict table
    print("\n=== 最终对照表 ===")
    print(f"{'method':<25} {'R@5':<10} {'R@10':<10} {'R@50':<10} {'R@100':<10} {'mean_rank':<10}")
    for name, m in summary.items():
        print(f"{name:<25} {m['R@5']:<10.6f} {m['R@10']:<10.6f} {m['R@50']:<10.6f} {m['R@100']:<10.6f} {m['mean_rank']:<10.1f}")

    print(f"\nTask #131 mean_pool R@5 = 0.00107")
    print(f"True random R@5 = {random_R[5]:.6f}")
    print(f"Last-item R@5 = {summary['last_item_only']['R@5']:.6f}")
    print(f"\n→ 如果 last-item R@5 也接近 0.00107, 说明 dense retrieval 在此 test 上确实无信号")
    print(f"→ 如果 last-item R@5 > mean_pool R@5 (例如 0.005), 说明 mean-pool 稀释了信号, 但仍然很弱")


if __name__ == "__main__":
    main()