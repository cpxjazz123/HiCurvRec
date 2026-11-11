"""Measurement primitives for the SID probes.

Occupancy, effective denominators and the two branch modes are separate numbers;
the earlier probe mixed them and reported a duplicate share as if it were the
largest bucket.
"""
from __future__ import annotations

import torch

LEVELS = ("L1", "L2", "L3")


def _bucket_sizes(keys) -> list[int]:
    counts: dict = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.values(), reverse=True)


def occupancy(tokens: torch.Tensor) -> dict:
    """Per-level prefix occupancy with the three shares kept apart."""
    n = int(tokens.shape[0])
    out = {}
    for level, name in enumerate(LEVELS):
        if tokens.shape[1] <= level:
            break
        counts: dict = {}
        for row in tokens[:, : level + 1].tolist():
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        buckets = sorted(counts.values(), reverse=True)
        duplicate = sum(size for size in buckets if size > 1) / max(n, 1)
        largest = buckets[0] / max(n, 1) if buckets else None
        out[name] = {
            "distinct_prefixes": len(counts),
            "buckets": buckets,
            "duplicate_item_share": duplicate,
            "max_bucket_share": largest,
            "one_minus_distinct": 1.0 - len(counts) / max(n, 1),
        }
    return out


def edge_stats(tokens: torch.Tensor, edge_index: torch.Tensor) -> dict:
    """Same-prefix share over the edges that actually fall inside the batch."""
    n = int(tokens.shape[0])
    kept = int(edge_index.shape[1]) if edge_index.numel() else 0
    out = {"kept_edges": kept}
    for level, name in enumerate(LEVELS):
        if tokens.shape[1] <= level:
            break
        entry = {"same_prefix_edges": 0, "same_prefix": "UNDEFINED",
                 "random_expectation": None}
        if kept:
            same = (tokens[edge_index[0], : level + 1]
                    == tokens[edge_index[1], : level + 1]).all(dim=1)
            entry["same_prefix_edges"] = int(same.sum())
            # the ratio is count / kept, not a float32 mean of the mask
            entry["same_prefix"] = entry["same_prefix_edges"] / kept
        counts: dict = {}
        for row in tokens[:, : level + 1].tolist():
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        if n > 1:
            total = sum(size * (size - 1) for size in counts.values())
            entry["random_expectation"] = total / (n * (n - 1))
        out[name] = entry
    return out


def branch_tokens(model, x: torch.Tensor, mode: str, infer_use_sk: bool = False):
    """Tokens from one branch, with the model mode restored afterwards."""
    if mode not in ("train", "export"):
        raise ValueError("mode must be train or export")
    was_training = model.training
    model.train(mode == "train")
    try:
        with torch.no_grad():
            tokens = model.get_indices(x, infer_use_sk=infer_use_sk)
    finally:
        model.train(was_training)
    return tokens.clone()
