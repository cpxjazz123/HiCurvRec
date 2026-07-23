"""
Task #68 — L1 品牌信息利用 step 1: L1 codeword 推断 + brand lookup table

输出:
  - products/task14/from_task68/l1_codeword.pt : (11924,) int64 L1 codeword per item
  - products/task14/from_task68/brand_lookup.json : L1 codeword (256) → mode brand name

启动:
  python scripts/task4_l1_brand_prep.py
"""

import json
import os
import sys

import numpy as np
import torch


def main():
    print("=" * 60)
    print("Task #68 — L1 codeword + brand lookup prep")
    print("=" * 60)

    out_dir = "products/task14/from_task68"
    os.makedirs(out_dir, exist_ok=True)

    # ============== 加载 RQ-VAE (Task 62 简化版) ==============
    print(f"\n[1/4] Loading Task 62 RQ-VAE...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task62_train_rqvae import ResidualQuantization

    ckpt = torch.load("products/task16/from_task62/rqvae_ckpt.pt",
                       map_location="cpu", weights_only=False)
    model = ResidualQuantization(
        n_features=ckpt["n_features"],
        n_layers=ckpt["n_layers"],
        n_clusters=ckpt["n_clusters"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ============== 加载 embeddings ==============
    print(f"\n[2/4] Loading embeddings...")
    embeddings = torch.load("products/task16/from_task62/item_embeddings.pt",
                             map_location="cpu", weights_only=False)
    print(f"  shape: {tuple(embeddings.shape)}")

    # ============== 推断 L1 codeword ==============
    print(f"\n[3/4] Inferring L1 codeword per item...")
    codebooks = [q.codebook for q in model.quantizers]
    cb_l1 = codebooks[0]

    with torch.no_grad():
        dist = torch.cdist(embeddings, cb_l1, p=2)
        l1_indices = dist.argmin(dim=1)
    print(f"  L1 codeword range: [{l1_indices.min().item()}, {l1_indices.max().item()}]")
    print(f"  L1 usage: {len(l1_indices.unique())} / {cb_l1.shape[0]} clusters used")

    # ============== 加载 metadata ==============
    print(f"\n[4/4] Loading metadata + building brand lookup...")
    metadata = torch.load("products/task16/from_task62/item_metadata.pt",
                          map_location="cpu", weights_only=False)

    # 提取 brand per item
    if isinstance(metadata, list):
        brands = [m.get("brand", "Unknown") if isinstance(m, dict) else "Unknown"
                  for m in metadata]
    else:
        brands = ["Unknown"] * len(metadata)

    print(f"  total items: {len(brands)}")
    print(f"  unique brands: {len(set(brands))}")

    # ============== Build L1 cluster → mode brand ==============
    n_clusters = cb_l1.shape[0]
    cluster_brand_count = [{} for _ in range(n_clusters)]  # cluster → {brand: count}
    for idx, brand in zip(l1_indices.tolist(), brands):
        if brand not in cluster_brand_count[idx]:
            cluster_brand_count[idx][brand] = 0
        cluster_brand_count[idx][brand] += 1

    # 每个 cluster 取众数 brand
    cluster_brand = {}
    cluster_brand_purity = {}
    for c in range(n_clusters):
        if cluster_brand_count[c]:
            mode_brand = max(cluster_brand_count[c], key=cluster_brand_count[c].get)
            cluster_brand[c] = mode_brand
            total = sum(cluster_brand_count[c].values())
            cluster_brand_purity[c] = cluster_brand_count[c][mode_brand] / total
        else:
            cluster_brand[c] = "Unknown"
            cluster_brand_purity[c] = 0.0

    # ============== 统计 ==============
    n_used = sum(1 for c in range(n_clusters) if cluster_brand_count[c])
    purity_mean = np.mean(list(cluster_brand_purity.values()))
    purity_used = [cluster_brand_purity[c] for c in range(n_clusters)
                   if cluster_brand_count[c]]
    purity_used_mean = np.mean(purity_used) if purity_used else 0.0

    n_unique_brand_per_cluster = len(set(cluster_brand.values()))
    print(f"  used clusters: {n_used} / {n_clusters}")
    print(f"  brand purity (used clusters mean): {purity_used_mean:.4f}")
    print(f"  unique brands after lookup: {n_unique_brand_per_cluster}")

    # ============== 保存 ==============
    print(f"\n  Saving...")
    torch.save(l1_indices, os.path.join(out_dir, "l1_codeword.pt"))

    lookup = {
        "cluster_to_brand": cluster_brand,
        "cluster_to_purity": cluster_brand_purity,
        "stats": {
            "n_clusters": n_clusters,
            "n_used_clusters": n_used,
            "n_unique_brands": len(set(brands)),
            "n_unique_brands_after_lookup": n_unique_brand_per_cluster,
            "purity_used_mean": float(purity_used_mean),
            "purity_all_mean": float(purity_mean),
        },
    }
    with open(os.path.join(out_dir, "brand_lookup.json"), "w") as f:
        json.dump(lookup, f, indent=2)

    # 输出 Top-10 高纯度 cluster（验证假设：L1 真的捕获品牌）
    sorted_clusters = sorted(
        [(c, cluster_brand_purity[c], cluster_brand_count[c][cluster_brand[c]])
         for c in range(n_clusters) if cluster_brand_count[c]],
        key=lambda x: -x[1]
    )[:10]
    print(f"\n  Top-10 cluster→brand purity (验证假设):")
    for c, purity, count in sorted_clusters:
        print(f"    cluster {c:3d}: brand={cluster_brand[c][:30]:30s} "
              f"purity={purity:.3f} count={count}")

    print(f"\n  Saved to {out_dir}/")
    print(f"    - l1_codeword.pt (11924 int64)")
    print(f"    - brand_lookup.json (256 cluster → mode brand)")


if __name__ == "__main__":
    main()