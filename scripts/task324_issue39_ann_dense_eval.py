#!/usr/bin/env python3
"""Task #324 / Issue #39 — Stage 4 召回改造 ANN dense retrieval 评估 (Arm A/B)

跟原 task84 evaluate() 协议不同 — 用 faiss HNSW/IVF-PQ 召回 item_emb (768d sentence-t5-base)
对 test sequences 做 dense retrieval top-K, 然后跟 ground truth item_id 比对 Recall@K.

协议差异:
- 原协议: T5.generate(num_beams=K) → K 个 SID 序列 → 查 SID→item_id 表 → Recall@K
- 新协议: query embedding (sequence mean pool) → faiss ANN → top-K items → Recall@K

两种协议产生不同 R@10 数字, 决策时必须分协议标注.
"""

import sys, os, json, time, argparse
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import numpy as np
import pandas as pd
import faiss
import torch
from torch.utils.data import DataLoader, Dataset


def load_item_embeddings(parquet_path):
    """Load (N, 768) item embeddings from parquet."""
    df = pd.read_parquet(parquet_path)
    item_ids = df['ItemID'].astype(str).tolist()
    embeddings = np.stack([np.array(e, dtype=np.float32) for e in df['embedding'].values])
    return item_ids, embeddings


def build_index(embeddings, arm, dim):
    """Build faiss index based on arm type."""
    if arm == 'A_hnsw':
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(embeddings)
    elif arm == 'B_ivf_pq':
        nlist, m, nbits = 64, 8, 8
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 8
    elif arm == 'E_dense':
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return index


def recall_at_k(pos_index, k):
    """pos_index: (batch, topK) bool. Recall@k = 1 if any top-K contains truth."""
    return pos_index[:, :k].any(dim=1).float()


def ndcg_at_k(pos_index, k):
    """NDCG@k: 1/log2(rank+2) if rank<k else 0."""
    # Find first true position
    pos_idx = pos_index.float()
    # Compute rank (1-indexed) of first true
    # pos_index[i, j] = True iff top-j candidate matches truth
    # First match rank = argmax of j where pos_index[i,j]=True, else K+1
    ranks = pos_idx.argmax(dim=1) + 1  # 1-indexed rank
    no_match = ~pos_idx.any(dim=1)
    ranks[no_match] = k + 1  # cap at K+1
    # NDCG = 1/log2(rank+1) if rank <= k else 0
    ndcg = torch.where(ranks <= k, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float))
    return ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, required=True, choices=['A_hnsw', 'B_ivf_pq', 'E_dense'])
    parser.add_argument('--topk_list', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--max_k', type=int, default=100, help='top-K from ANN (≥ max(topk_list))')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Load item embeddings
    emb_path = 'HG-Rec/dataset/Instruments/item_emb.parquet'
    item_ids, embeddings = load_item_embeddings(emb_path)
    print(f'[task324-ANN/{args.arm}] Items: {len(item_ids)}, Emb dim: {embeddings.shape[1]}', flush=True)

    # L2 normalize for cosine
    faiss.normalize_L2(embeddings)

    # Build faiss index
    index = build_index(embeddings, args.arm, embeddings.shape[1])
    print(f'[task324-ANN/{args.arm}] Built {type(index).__name__}, ntotal={index.ntotal}', flush=True)

    # Build item_id → index mapping
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    # Load test dataset
    test_df = pd.read_parquet('HG-Rec/dataset/Instruments/test.parquet')
    print(f'[task324-ANN/{args.arm}] Test set: {len(test_df)} sequences', flush=True)

    # For each sequence: query = mean of history item embeddings, ground truth = target item_id
    t0 = time.time()
    recalls_all = {f'Recall@{k}': [] for k in args.topk_list}
    ndcgs_all = {f'NDCG@{k}': [] for k in args.topk_list}

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        history_items = row['history']  # list of item_ids
        target_item = row['target']      # ground truth item_id (single)

        # Get history item embeddings (test history is numpy int64, emb ItemID is str)
        hist_indices = [item_id_to_idx.get(str(int(h))) for h in history_items]
        hist_indices = [i for i in hist_indices if i is not None]
        if not hist_indices:
            continue
        if not hist_indices:
            continue
        hist_emb = embeddings[hist_indices]
        # Mean pool as query
        query = hist_emb.mean(axis=0, keepdims=True).astype('float32')
        faiss.normalize_L2(query)

        # Top-K retrieve
        D, I = index.search(query, args.max_k)
        retrieved_items = [item_ids[i] for i in I[0]]
        # Check ground truth in top-K (target is numpy.int64, convert to str)
        target_str = str(int(target_item))
        pos_index = torch.tensor([target_str == ri for ri in retrieved_items], dtype=torch.bool).unsqueeze(0)

        for k in args.topk_list:
            recalls_all[f'Recall@{k}'].append(recall_at_k(pos_index, k).item())
            ndcgs_all[f'NDCG@{k}'].append(ndcg_at_k(pos_index, k).item())

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            r10 = sum(recalls_all['Recall@10']) / len(recalls_all['Recall@10'])
            print(f'[task324-ANN/{args.arm}] {idx+1}/{len(test_df)} elapsed={elapsed:.1f}s R@10={r10:.4f}', flush=True)

    elapsed = time.time() - t0
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls_all.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs_all.items()}

    out = {
        'task': f'task324_issue39_ann_arm_{args.arm}',
        'arm': args.arm,
        'protocol': 'ANN_dense_retrieval',  # 跟 T5.generate SID 协议区分
        'item_count': len(item_ids),
        'top_k_max': args.max_k,
        'elapsed_sec': elapsed,
        **{f'test_{k}': v for k, v in avg_recalls.items()},
        **{f'test_{k}': v for k, v in avg_ndcgs.items()},
    }
    metrics_dir = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts'
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, f'task324_ann_arm_{args.arm}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'[task324-ANN/{args.arm}] DONE elapsed={elapsed:.1f}s', flush=True)
    print(f'  R@5/10/20 = [{avg_recalls["Recall@5"]:.4f}, {avg_recalls["Recall@10"]:.4f}, {avg_recalls["Recall@20"]:.4f}]')
    print(f'  NDCG@5/10/20 = [{avg_ndcgs["NDCG@5"]:.4f}, {avg_ndcgs["NDCG@10"]:.4f}, {avg_ndcgs["NDCG@20"]:.4f}]')
    print(f'  Metrics written to {out_path}')


if __name__ == '__main__':
    main()