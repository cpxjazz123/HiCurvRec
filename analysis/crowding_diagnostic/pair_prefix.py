"""Pair-level preference and longest-common-prefix classification."""
from __future__ import annotations

import numpy as np


CLASS_NAMES = (
    "positive_prefix_advantage",
    "negative_prefix_advantage",
    "both_share",
    "neither_shares",
)


def classify_prefix(positive_shares, negative_shares) -> dict:
    """Classify each anchor-positive-matched-negative set into one of four outcomes."""
    positive = np.asarray(positive_shares, dtype=bool).reshape(-1)
    negative = np.asarray(negative_shares, dtype=bool)
    if negative.ndim == 1:
        negative = negative[:, None]
    if negative.ndim != 2 or negative.shape[0] != positive.size:
        raise ValueError("positive and negative sharing arrays must align by anchor")
    any_negative = negative.any(axis=1)
    classes = {
        "positive_prefix_advantage": positive & ~any_negative,
        "negative_prefix_advantage": ~positive & any_negative,
        "both_share": positive & any_negative,
        "neither_shares": ~positive & ~any_negative,
    }
    total = positive.size
    result = {name: mask for name, mask in classes.items()}
    counts = {name: int(mask.sum()) for name, mask in classes.items()}
    rates = {name: (count / total if total else None) for name, count in counts.items()}
    result.update({
        "counts": counts,
        "rates": rates,
        "total": int(total),
        "positive_advantage_count": counts["positive_prefix_advantage"],
        "negative_advantage_count": counts["negative_prefix_advantage"],
        "tie_count": counts["both_share"] + counts["neither_shares"],
        "positive_prefix_advantage_rate": rates["positive_prefix_advantage"],
        "negative_prefix_advantage_rate": rates["negative_prefix_advantage"],
        "both_share_rate": rates["both_share"],
        "neither_shares_rate": rates["neither_shares"],
        "tie_rate": ((counts["both_share"] + counts["neither_shares"]) / total if total else None),
        "preference_accuracy": (
            counts["positive_prefix_advantage"] /
            (counts["positive_prefix_advantage"] + counts["negative_prefix_advantage"])
            if counts["positive_prefix_advantage"] + counts["negative_prefix_advantage"] else None
        ),
    })
    return result


def longest_common_prefix(left, right, codebook_tokens: int = 3):
    """Return row-wise LCP lengths, excluding the deterministic collision extension."""
    a = np.asarray(left)[:, :codebook_tokens]
    b = np.asarray(right)[:, :codebook_tokens]
    if a.shape != b.shape:
        raise ValueError("LCP inputs must have equal shape")
    equal = a == b
    mismatch = np.argmax(~equal, axis=1)
    return np.where(equal.all(axis=1), a.shape[1], mismatch).astype(np.int8)


def _lcp_arrays(positive_lcp, negative_lcp):
    positive = np.asarray(positive_lcp).reshape(-1)
    negative = np.asarray(negative_lcp)
    if negative.ndim == 1:
        negative = negative[:, None]
    if negative.ndim != 2 or negative.shape[0] != positive.size:
        raise ValueError("positive and negative LCP arrays must align by anchor")
    return positive[:, None], negative


def lcp_pairwise_auc(positive_lcp, negative_lcp) -> float:
    """Positive-favoured AUC over every matched negative comparison."""
    positive, negative = _lcp_arrays(positive_lcp, negative_lcp)
    return float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative))


def lcp_max_auc(positive_lcp, negative_lcp) -> float:
    """Positive-favoured AUC against each anchor's hardest (maximum-LCP) negative."""
    positive, negative = _lcp_arrays(positive_lcp, negative_lcp)
    hardest_negative = negative.max(axis=1, keepdims=True)
    return float(np.mean(positive > hardest_negative) +
                 0.5 * np.mean(positive == hardest_negative))


def classify_lcp(positive_lcp, negative_lcp):
    """Classify LCP ordering; ties split by whether the prefix is shared at all."""
    positive = np.asarray(positive_lcp).reshape(-1)
    negative = np.asarray(negative_lcp)
    if negative.ndim == 1:
        negative = negative[:, None]
    if negative.ndim != 2 or negative.shape[0] != positive.size:
        raise ValueError("positive and negative LCP arrays must align by anchor")
    max_negative = negative.max(axis=1)
    greater = positive > max_negative
    less = positive < max_negative
    return {
        "positive_prefix_advantage": greater,
        "negative_prefix_advantage": less,
        "both_share": ~(greater | less) & (positive > 0),
        "neither_shares": ~(greater | less) & (positive == 0),
    }
