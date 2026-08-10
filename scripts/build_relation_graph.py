#!/usr/bin/env python3
"""Issue #118 Phase B Task 1 (2026-08-10): Build Frozen Item Relation Graph.

使用 Stage1 frozen item embedding, 基于 cosine similarity 建 KNN graph.

对每个 item (i):
  - positive: j+ ∈ TopKNN_PosK(i)         (默认 POS_K=8)
  - negative: j- ∉ TopKNN_NegExcl(i)      (默认 NEG_EXCL=64, 排除大近邻集合)
                + 随机 sample NEG_N 个     (默认 NEG_N=32)

输出:
  - relation_graph.npz:
      pos_idx: (N_ITEMS, POS_K) int32
      neg_idx: (N_ITEMS, NEG_N) int32
  - relation_graph_metadata.json:
      embedding_sha256, similarity=cosine, POS_K, NEG_N, NEG_EXCL,
      n_items, seed, build_time, statistics

Relation graph 在训练开始前冻结. 禁止随 curvature/encoder 更新重新生成 KNN,
否则 target 本身会跟模型一起漂移,无法成为真正的 relational objective.
"""
import sys
import os
import json
import time
import hashlib
import argparse
from pathlib import Path

import numpy as np

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, GENRE_ROOT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Issue #118 Build Frozen Item Relation Graph")
    parser.add_argument("--item_emb_npy", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue96_v74_repro/item_emb_baseline_u32.npy")
    parser.add_argument("--pos_k", type=int, default=8)
    parser.add_argument("--neg_n", type=int, default=32)
    parser.add_argument("--neg_excl", type=int, default=64,
                        help="exclude Top{neg_excl} neighbors when sampling negatives")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_relational_smoke")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rel_graph] loading item_emb from {args.item_emb_npy}")
    item_emb = np.load(args.item_emb_npy, allow_pickle=True).astype(np.float32)
    assert item_emb.ndim == 2, f"item_emb must be 2D, got {item_emb.shape}"
    N, D = item_emb.shape
    print(f"[rel_graph] item_emb shape={item_emb.shape}")

    # 1. L2 normalize for cosine similarity
    norms = np.linalg.norm(item_emb, axis=1, keepdims=True).clip(min=1e-12)
    emb_normed = item_emb / norms
    print(f"[rel_graph] L2 normalized, computing cosine sim matrix ({N}x{N})...")

    t0 = time.time()
    # 2. Cosine similarity matrix (N x N) — float32 OK
    # Use chunked to avoid OOM for very large N
    chunk = 1024
    top_k = args.neg_excl + 1  # +1 to exclude self
    pos_idx_full = np.zeros((N, args.pos_k), dtype=np.int32)
    neg_idx_full = np.zeros((N, args.neg_n), dtype=np.int32)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sim_chunk = emb_normed[start:end] @ emb_normed.T  # (chunk, N)
        # 取 Top(neg_excl+1) 用于负样本排除
        top_idx = np.argpartition(-sim_chunk, top_k, axis=1)[:, :top_k]  # (chunk, top_k)
        # 按相似度排序
        for i_local, i_global in enumerate(range(start, end)):
            cand_idx = top_idx[i_local]
            cand_sim = sim_chunk[i_local, cand_idx]
            sort_order = np.argsort(-cand_sim)
            sorted_idx = cand_idx[sort_order]
            # 排除 self (i_global 可能在 sorted_idx 里)
            mask = sorted_idx != i_global
            sorted_idx = sorted_idx[mask]
            # 前 POS_K 作正样本
            pos_idx_full[i_global, :args.pos_k] = sorted_idx[:args.pos_k]
            # 从 sorted_idx[POS_K:neg_excl] 之外随机 NEG_N 个 (排除近邻大集合)
            neg_pool_start = args.neg_excl
            neg_pool = np.arange(N)
            # 排除 sorted_idx[:neg_excl] (近邻大集合)
            exclude_set = set(sorted_idx[:args.neg_excl].tolist())
            # 同时排除 i_global 自身
            exclude_set.add(i_global)
            neg_pool_mask = np.ones(N, dtype=bool)
            for ex in exclude_set:
                neg_pool_mask[ex] = False
            neg_pool = neg_pool[neg_pool_mask]
            if len(neg_pool) < args.neg_n:
                # 兜底: 全部可用负样本
                neg_idx_full[i_global, :len(neg_pool)] = neg_pool
                # 剩余位置随机采
                extra = np.random.choice(N, args.neg_n - len(neg_pool), replace=True)
                neg_idx_full[i_global, len(neg_pool):] = extra
            else:
                neg_idx_full[i_global] = np.random.choice(neg_pool, args.neg_n, replace=False)
        if start % (chunk * 10) == 0:
            print(f"[rel_graph]   processed {end}/{N} items ({time.time() - t0:.1f}s)")

    t1 = time.time()
    print(f"[rel_graph] DONE in {t1 - t0:.1f}s")

    # 3. 保存
    rel_graph_path = output_dir / "relation_graph.npz"
    np.savez_compressed(
        rel_graph_path,
        pos_idx=pos_idx_full,
        neg_idx=neg_idx_full,
    )
    print(f"[rel_graph] saved {rel_graph_path} (pos_idx {pos_idx_full.shape}, neg_idx {neg_idx_full.shape})")

    # 4. metadata
    emb_sha = sha256_file(args.item_emb_npy)
    metadata = {
        "issue": "#118",
        "task": "#1",
        "title": "Frozen Item Relation Graph (KNN cosine)",
        "embedding_source": args.item_emb_npy,
        "embedding_sha256": emb_sha,
        "embedding_shape": list(item_emb.shape),
        "embedding_dtype": "float32",
        "similarity_metric": "cosine",
        "pos_k": args.pos_k,
        "neg_n": args.neg_n,
        "neg_excl": args.neg_excl,
        "n_items": int(N),
        "seed": args.seed,
        "build_time_sec": t1 - t0,
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "statistics": {
            "pos_idx_mean": float(pos_idx_full.mean()),
            "pos_idx_max": int(pos_idx_full.max()),
            "neg_idx_mean": float(neg_idx_full.mean()),
            "neg_idx_max": int(neg_idx_full.max()),
            # self-exclusion sanity check: pos_idx[i, 0] != i for all i
            "pos_self_exclusion_ok": bool(np.all(pos_idx_full[:, 0] != np.arange(N))),
        },
    }
    meta_path = output_dir / "relation_graph_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[rel_graph] saved {meta_path}")

    # sanity 校验
    print(f"\n[rel_graph] sanity:")
    print(f"  pos_idx shape: {pos_idx_full.shape}, dtype: {pos_idx_full.dtype}")
    print(f"  neg_idx shape: {neg_idx_full.shape}, dtype: {neg_idx_full.dtype}")
    print(f"  pos self-exclusion ok: {metadata['statistics']['pos_self_exclusion_ok']}")
    # 检查 negative 不会落在 pos 内
    overlap = 0
    for i in range(min(100, N)):
        overlap += len(set(pos_idx_full[i].tolist()) & set(neg_idx_full[i].tolist()))
    print(f"  pos-neg overlap (first 100 items): {overlap}/{(min(100, N)) * args.pos_k} (期望接近 0)")

    print(f"\n[rel_graph] DONE. artifacts in {output_dir}/")


if __name__ == "__main__":
    main()