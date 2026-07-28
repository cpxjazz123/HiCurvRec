"""Task #180 Stage 0 — 共现图 G2 重建 (LightGCN Step 1).

从 Instruments train.parquet 构建 user→[items] 序列,
按 window=5 双方向生成 item-item 共现边 (跟 Task #69 build_G2 同款 recipe).

输出:
  - products/task180/graph/G2_cooccurrence_kg_final.txt  (h, r, t 三元组)
  - products/task180/graph/G2_cooccurrence_adj.npy       (N, N) scipy.sparse adjacency
  - products/task180/graph/item_id_mapping.parquet       (item 原 ID → embedding index 0..N-1)
"""
import argparse
import collections
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task180_build_G2")


def load_train_sequences(parquet_path: str) -> dict:
    """Read train.parquet → {user_id: [item_id_1, item_id_2, ...]}.

    HG-Rec Instruments schema: user (int), history (list[int]), target (int).
    Each row → user, history+target 作为该用户的完整 item 序列.
    """
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df)} interactions from {parquet_path}")
    log.info(f"  Columns: {df.columns.tolist()}")

    if 'user' not in df.columns or 'history' not in df.columns:
        raise ValueError(
            f"Expected HG-Rec schema (user, history, target), got {df.columns.tolist()}"
        )

    user_seqs = collections.defaultdict(list)
    n_users = 0
    for u, hist, tgt in zip(df['user'].values, df['history'].values, df['target'].values):
        # 同一 user 可能有多行 (不同时间窗), history 已经按时间排好
        # 我们把 history + target 作为该 user 的 item 序列
        # (target 不在 history 里, 但作为后续上下文, 应该被加入序列)
        # 如果 user 已经存在, 拼接; 否则新建
        if u in user_seqs:
            user_seqs[u].extend(hist)
            user_seqs[u].append(tgt)
        else:
            user_seqs[u] = list(hist) + [tgt]
            n_users += 1

    # 验证 seq 长度 >=2 (否则 build_cooccurrence_edges 会跳过)
    n_skipped_short = 0
    final_seqs = {}
    for u, items in user_seqs.items():
        if len(items) < 2:
            n_skipped_short += 1
            continue
        final_seqs[u] = items

    log.info(f"  Total users: {len(final_seqs)}, total items in sequences: {len(set(i for s in final_seqs.values() for i in s))}")
    log.info(f"  Skipped {n_skipped_short} users with <2 items")

    return final_seqs


def build_cooccurrence_edges(user_seqs: dict, window: int = 5) -> set:
    """按 window 内前后 item 配对, 生成 (item_a, item_b) 边集 (双方向)."""
    edges = set()
    for u, items in tqdm(user_seqs.items(), desc=f"Building G2 (window={window})"):
        n = len(items)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, min(i + window, n)):
                if items[i] != items[j]:
                    a, b = items[i], items[j]
                    edges.add((a, b))  # 无向: 加 (a, b) 即可 (后续 adjacency 自动对称)
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet")
    parser.add_argument("--item_emb_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task180/graph")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load train sequences
    user_seqs = load_train_sequences(args.train_path)

    # 2. Load item embedding to get item-id ordering
    item_emb_df = pd.read_parquet(args.item_emb_path)
    log.info(f"Loaded {len(item_emb_df)} items from item_emb.parquet")

    # 跟 task180/train.parquet / item_emb.parquet 对齐: item_emb.parquet 已经有 9922 items
    # 我们构建 embedding_index → item_id 的映射
    if 'item_id' in item_emb_df.columns:
        emb_items = item_emb_df['item_id'].values.tolist()
    else:
        # 无 item_id 列, 默认顺序 0..N-1
        emb_items = list(range(len(item_emb_df)))

    item_id_to_emb_idx = {iid: idx for idx, iid in enumerate(emb_items)}
    n_items = len(emb_items)
    log.info(f"  N items in embedding space: {n_items}")

    # 3. Build cooccurrence edges
    edges = build_cooccurrence_edges(user_seqs, args.window)
    log.info(f"  Total unique edges (undirected): {len(edges)}")

    # 4. Filter edges to only those where both endpoints are in item_emb
    edges_in_vocab = []
    skipped = 0
    for a, b in edges:
        if a in item_id_to_emb_idx and b in item_id_to_emb_idx:
            edges_in_vocab.append((item_id_to_emb_idx[a], item_id_to_emb_idx[b]))
        else:
            skipped += 1
    log.info(f"  Edges in vocab: {len(edges_in_vocab)} (skipped {skipped} out-of-vocab)")

    # 5. Write kg_final.txt (跟 MCKG format 一致: h r t)
    kg_path = os.path.join(args.output_dir, "G2_cooccurrence_kg_final.txt")
    with open(kg_path, 'w') as f:
        for h, t in edges_in_vocab:
            f.write(f"{h}\t0\t{t}\n")
    log.info(f"  Wrote: {kg_path} ({len(edges_in_vocab)} triples)")

    # 6. Build sparse adjacency matrix (N x N symmetric)
    import scipy.sparse as sp
    rows = [h for h, t in edges_in_vocab] + [t for h, t in edges_in_vocab]
    cols = [t for h, t in edges_in_vocab] + [h for h, t in edges_in_vocab]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_items, n_items))
    adj = adj.tocsr()

    # 7. Compute D^{-1/2} A D^{-1/2} (LightGCN symmetric normalization)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-10), -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    adj_norm = adj_norm.tocsr()

    log.info(f"  Adjacency: shape {adj.shape}, nnz = {adj.nnz}, avg degree = {deg.mean():.2f}")

    adj_path = os.path.join(args.output_dir, "G2_cooccurrence_adj_norm.npz")
    sp.save_npz(adj_path, adj_norm)
    log.info(f"  Wrote normalized adjacency: {adj_path}")

    # 8. Save item_id mapping (in case future code needs raw item_id ↔ emb_idx)
    mapping_df = pd.DataFrame({
        'item_id': emb_items,
        'emb_idx': list(range(n_items)),
    })
    mapping_path = os.path.join(args.output_dir, "item_id_mapping.parquet")
    mapping_df.to_parquet(mapping_path)
    log.info(f"  Wrote item_id mapping: {mapping_path}")

    # 9. Diagnostic summary
    diag = {
        "n_interactions": int(sum(len(v) for v in user_seqs.values())),
        "n_users": len(user_seqs),
        "n_items_in_emb": n_items,
        "n_unique_edges_undirected": len(edges_in_vocab),
        "window": args.window,
        "avg_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "min_degree": int(deg[deg > 0].min()) if (deg > 0).any() else 0,
        "skipped_out_of_vocab": skipped,
    }
    diag_path = os.path.join(args.output_dir, "G2_cooccurrence_diagnostic.json")
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    log.info(f"  Wrote diagnostic: {diag_path}")
    log.info(f"  G2 graph build COMPLETE — {diag}")


if __name__ == '__main__':
    main()