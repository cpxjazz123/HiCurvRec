#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #94 Borda rank fusion ensemble (2026-08-09) - common/ 复用脚本.

把多个 ckpt 给出的 top-K predictions 聚合, 用 Borda 计数 (排名分数求和) 重排, 重新计算 R@K.

输入: N 个 raw_predictions.npz 文件 (从 stage4_eval_pure_t5_v85p_4layer.py 落盘)
输出: 聚合后的 R@5/R@10/R@20/NDCG@{5,10,20} metrics

用法:
  python3 common/ensemble/borda_rank_fusion.py \\
    --pred_npys /tmp/v85p_ep130_b100/raw_predictions.npz /tmp/v85p_ep150_b50/raw_predictions.npz ... \\
    --weights 3.0 2.0 2.0 \\
    --out /tmp/ensemble_result.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def dcg(rel, k):
    return sum(rel[i] / math.log2(i + 2) for i in range(min(k, len(rel))))


def evaluate(ranked_per_user, labels, ks=(5, 10, 20)):
    n = len(ranked_per_user)
    hits = {f"R@{k}": 0 for k in ks}
    ndcgs = {f"NDCG@{k}": 0.0 for k in ks}
    for u, ranked in enumerate(ranked_per_user):
        target = tuple(labels[u].tolist())
        rels = [1 if tuple(sid) == target else 0 for sid in ranked]
        for k in ks:
            topk_rels = rels[:k]
            hits[f"R@{k}"] += sum(topk_rels)
            ndcgs[f"NDCG@{k}"] += dcg(topk_rels, k) / (dcg([1], k) if k > 0 else 1)
    result = {k: v / n for k, v in hits.items()}
    result.update({k: v / n for k, v in ndcgs.items()})
    result["n_eval"] = n
    return result


def borda_rank_fusion(preds_list, weights):
    """Borda rank fusion: rank sum weighted.

    Args:
        preds_list: list of np.ndarray (n, beam_i, 4) — top-K predictions per ckpt
        weights: list of float — per-ckpt weights (e.g. [3.0, 2.0, 2.0])

    Returns:
        ranked_per_user: list of lists of SID tuples (ranked by Borda score desc)
    """
    n = preds_list[0].shape[0]
    beams = [p.shape[1] for p in preds_list]
    preds_tuples = [[[tuple(s) for s in preds_list[i][u].tolist()] for u in range(n)] for i in range(len(preds_list))]
    out = []
    for u in range(n):
        scores = {}
        for i, w in enumerate(weights):
            beam_i = beams[i]
            for rank, sid in enumerate(preds_tuples[i][u]):
                scores[sid] = scores.get(sid, 0) + w * (beam_i - rank)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out.append([sid for sid, _ in ranked])
    return out


def reciprocal_rank_fusion(preds_list, weights, k=60):
    """RRF: 1/(k+rank) score fusion (Cormack 2009)."""
    n = preds_list[0].shape[0]
    preds_tuples = [[[tuple(s) for s in preds_list[i][u].tolist()] for u in range(n)] for i in range(len(preds_list))]
    out = []
    for u in range(n):
        scores = {}
        for i, w in enumerate(weights):
            for rank, sid in enumerate(preds_tuples[i][u]):
                scores[sid] = scores.get(sid, 0) + w / (k + rank)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out.append([sid for sid, _ in ranked])
    return out


def main():
    parser = argparse.ArgumentParser(description="Borda/RRF rank fusion ensemble (Issue #94)")
    parser.add_argument("--pred_npys", nargs="+", required=True, help="raw_predictions.npz 文件路径列表")
    parser.add_argument("--weights", nargs="+", type=float, required=True, help="per-ckpt 权重")
    parser.add_argument("--method", choices=["borda", "rrf"], default="borda")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter (only for rrf method)")
    parser.add_argument("--out", type=str, required=True, help="输出 verdict JSON 路径")
    args = parser.parse_args()

    assert len(args.pred_npys) == len(args.weights), f"pred_npys ({len(args.pred_npys)}) vs weights ({len(args.weights)}) count mismatch"

    print(f"[Borda] loading {len(args.pred_npys)} ckpts...")
    data = [np.load(p) for p in args.pred_npys]
    preds_list = [d["preds"] for d in data]
    labels = data[0]["labels"]
    beams = [p.shape[1] for p in preds_list]
    print(f"[Borda] beams per ckpt: {beams}, total n={preds_list[0].shape[0]}")

    if args.method == "borda":
        ranked = borda_rank_fusion(preds_list, args.weights)
    else:
        ranked = reciprocal_rank_fusion(preds_list, args.weights, k=args.rrf_k)

    result = evaluate(ranked, labels)
    result["method"] = args.method
    result["weights"] = args.weights
    result["beams"] = beams
    result["pred_npys"] = args.pred_npys

    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Borda] saved {out_path}")


if __name__ == "__main__":
    main()