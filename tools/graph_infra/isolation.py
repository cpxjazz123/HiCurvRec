"""Held-out blocking and bank isolation (issue #246 C5-C7)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def held_out_support(visible: sp.spmatrix, evaluation: sp.spmatrix) -> sp.csr_matrix:
    """Undirected support of edges present in the evaluation graph but absent from the visible one.

    Weighted subtraction is wrong: a larger visible weight would cancel a real held-out direction.
    The support is computed on booleans, then symmetrised.
    """
    support_v = (visible > 0).astype(np.int8).tocsr()
    support_e = (evaluation > 0).astype(np.int8).tocsr()
    difference = support_e - support_e.multiply(support_v).astype(np.int8)
    difference.data = (difference.data > 0).astype(np.int8)
    difference.eliminate_zeros()
    symmetric = (difference + difference.T) > 0
    return symmetric.astype(np.int8).tocsr()


def pair_keys(pairs) -> set:
    """Undirected canonical keys of a pair array."""
    return {(int(min(a, b)), int(max(a, b))) for a, b in np.asarray(pairs, dtype=np.int64).tolist()}


def blocked_pairs(pairs, held: sp.spmatrix) -> list:
    """Pairs whose undirected key is blocked in the held-out support (both directions tested)."""
    bad = []
    for left, right in np.asarray(pairs, dtype=np.int64).tolist():
        if held[left, right] > 0 or held[right, left] > 0:
            bad.append((left, right))
    return bad


def audit_banks(train_banks: dict, strict_eval_banks: dict, held: sp.spmatrix) -> dict:
    """Every invariant the issue asks for, measured on actual pair sets."""
    train_keys = set()
    per_group = {}
    for name, pairs in train_banks.items():
        keys = pair_keys(pairs)
        per_group[name] = {
            "pairs": len(pairs),
            "distinct_undirected": len(keys),
            "blocked": len(blocked_pairs(pairs, held)),
        }
        train_keys |= keys
    eval_keys = {}
    for name, pairs in strict_eval_banks.items():
        keys = pair_keys(pairs)
        eval_keys[name] = keys
    overlaps = {name: len(keys & train_keys) for name, keys in eval_keys.items()}
    cross = {}
    names = list(eval_keys)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            cross[f"{left}&{right}"] = len(eval_keys[left] & eval_keys[right])
    return {
        "train": per_group,
        "train_blocked_total": sum(group["blocked"] for group in per_group.values()),
        "eval_vs_train_overlap": overlaps,
        "eval_cross_overlap": cross,
        "ok": (sum(group["blocked"] for group in per_group.values()) == 0
               and all(value == 0 for value in overlaps.values())),
    }
