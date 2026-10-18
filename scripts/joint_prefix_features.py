"""Stage 2 held-out feature extraction for issue #47.

Computes the frozen-valid-side behavioural and structural features for every model in
``model_inventory.json``, on the same 52,855 held-out transitions x 10 matched negatives.
Definitions are identical to the two existing read-only implementations this issue
points at (``analyze_heldout_prefix_enrichment.py`` and ``analyze_hbpe_r2_enrichment.py``)
and are NOT changed here:

    pos_share_l = mean(prefix_l(anchor) == prefix_l(positive))
    neg_share_l = mean(prefix_l(anchor) == prefix_l(negative))
    gap_l       = pos_share_l - neg_share_l
    ER_l        = pos_share_l / neg_share_l      (only when the denominator > 0)

The 4th token is the collision-extension token: it takes no part in any L1 / L1+L2
prefix comparison, and ``full_3token_collision`` is built from the first three tokens
only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_PAIRS_PATH = (
    REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
)
SID_ROOT_OVERRIDES = [Path("/tmp/r3_recover")]

BASELINE_L1_ER = 16.38953942232631
BASELINE_L1L2_ER = 58.38709677419355


def resolve_sid_path(sid_path: str) -> Path | None:
    """Inventory path first, then the read-only staging root used for recovered SIDs."""
    direct = REPO_ROOT / sid_path
    if direct.is_file():
        return direct
    for root in SID_ROOT_OVERRIDES:
        candidate = root / sid_path
        if candidate.is_file():
            return candidate
    return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sid_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (matrix[n_items, 4], audit). Row index is the raw item id."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    sids = json.loads(raw.decode("utf-8"))
    n_items = len(sids)
    item_ids = sorted(int(k) for k in sids)
    max_id = item_ids[-1]
    matrix = np.full((max_id + 1, 4), -1, dtype=np.int64)
    widths = set()
    coverage_gaps = []
    prev = -1
    for iid in item_ids:
        row = sids[str(iid)]
        widths.add(len(row))
        if prev >= 0 and iid != prev + 1:
            coverage_gaps.append((prev + 1, iid - 1))
        prev = iid
        matrix[iid] = row
    audit = {
        "sid_sha256": sha,
        "n_items": n_items,
        "min_item_id": item_ids[0],
        "max_item_id": item_ids[-1],
        "row_widths": sorted(widths),
        "coverage_gaps": len(coverage_gaps),
        "all_rows_filled": bool((matrix[item_ids] >= 0).all()),
        "collision_token_present": bool(widths == {4}),
    }
    return matrix, audit


def prefix_match(
    matrix: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    if depth == 1:
        a, p, n = matrix[anchors, 0], matrix[positives, 0], matrix[negatives, 0]
        return a == p, a[:, None] == n
    a, p, n = matrix[anchors, :2], matrix[positives, :2], matrix[negatives, :2]
    return np.all(a == p, axis=1), np.all(a[:, None, :] == n, axis=2)


def codebook_structure(matrix: np.ndarray, capacity: int = 256) -> dict[str, Any]:
    """L1 usage from the true hard codes of every catalogue item, never from pairs."""
    tokens = matrix[:, 0]
    valid = tokens[tokens >= 0]
    counts = np.bincount(valid, minlength=capacity).astype(np.float64)
    share = counts / counts.sum()
    nonzero = share[share > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return {
        "l1_distinct_codes": int((counts > 0).sum()),
        "l1_codebook_utilisation": float((counts > 0).sum()) / capacity,
        "l1_entropy_nats": entropy,
        "l1_effective_usage_exp_H": float(np.exp(entropy)),
        "l1_effective_usage_ratio": float(np.exp(entropy)) / capacity,
        "l1_largest_group_share": float(share.max()),
        "l1_zero_code_count": int((counts == 0).sum()),
    }


def level_features(
    matrix: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    depth: int,
    name: str,
) -> dict[str, Any]:
    pos_match, neg_match = prefix_match(matrix, anchors, positives, negatives, depth)
    pos_count = int(pos_match.sum())
    neg_count = int(neg_match.sum())
    n_pos = int(pos_match.size)
    n_neg = int(neg_match.size)
    pos_share = pos_count / n_pos
    neg_share = neg_count / n_neg
    er: float | str
    if neg_count == 0:
        er = "undefined_zero_neg_share"
    elif pos_share == 1.0 and neg_share == 1.0:
        er = 1.0
    else:
        er = float(pos_share / neg_share)
    return {
        "level": name,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "n_pos_draws": n_pos,
        "n_neg_draws": n_neg,
        "pos_share": float(pos_share),
        "neg_share": float(neg_share),
        "gap": float(pos_share - neg_share),
        "er": er,
        "er_defined": neg_count > 0,
        "pos_match": pos_match,
        "neg_match": neg_match,
    }


def structural_features(matrix: np.ndarray, capacity: int = 256) -> dict[str, Any]:
    out = codebook_structure(matrix, capacity)
    out["l12_distinct_prefixes"] = int(len(np.unique(matrix[:, :2], axis=0)))
    valid_rows = matrix[:, 0] >= 0
    triples = matrix[valid_rows, :3]
    unique_triples = int(len(np.unique(triples, axis=0)))
    n_valid = int(valid_rows.sum())
    out["l123_distinct_triples"] = unique_triples
    out["full_3token_collision"] = float(1.0 - unique_triples / n_valid)
    out["n_items_scored"] = n_valid
    return out


def clean(level: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in level.items() if not isinstance(v, np.ndarray)}


def canonical_check(l1: dict[str, Any], l1l2: dict[str, Any]) -> dict[str, Any]:
    l1_er = l1["er"]
    l2_er = l1l2["er"]
    return {
        "l1_er_measured": l1_er,
        "l1_er_expected": BASELINE_L1_ER,
        "l1_abs_error": (
            abs(float(l1_er) - BASELINE_L1_ER) if isinstance(l1_er, float) else None
        ),
        "l1l2_er_measured": l2_er,
        "l1l2_er_expected": BASELINE_L1L2_ER,
        "l1l2_abs_error": (
            abs(float(l2_er) - BASELINE_L1L2_ER) if isinstance(l2_er, float) else None
        ),
    }


def extract(model: dict[str, Any], anchors, positives, negatives) -> dict[str, Any]:
    path = resolve_sid_path(model["sid_path"])
    if path is None:
        return {"model": model["model"], "error": "SID file not found", "sid_path": model["sid_path"]}
    matrix, audit = load_sid_matrix(path)
    if audit["row_widths"] != [4]:
        return {"model": model["model"], "error": f"unexpected token width {audit['row_widths']}",
                "sid_path": model["sid_path"]}
    if not audit["sid_sha256"].startswith(model["sid_sha256"]):
        return {"model": model["model"], "error": "SID sha256 does not match inventory",
                "sid_path": model["sid_path"], "measured": audit["sid_sha256"],
                "expected": model["sid_sha256"]}

    l1 = level_features(matrix, anchors, positives, negatives, 1, "L1")
    l1l2 = level_features(matrix, anchors, positives, negatives, 2, "L1L2")
    struct = structural_features(matrix)

    row: dict[str, Any] = {
        "model": model["model"],
        "family": model["family"],
        "sid_path": model["sid_path"],
        "sid_recovered_from_git": not (REPO_ROOT / model["sid_path"]).is_file(),
        "stage3_recall@10": model["recall@10"],
        "stage3_ndcg@10": model["ndcg@10"],
        "stage3_n_eval": model["n_eval"],
        "stage3_n_runs": model["n_stage3_runs"],
        "sid_sha256": audit["sid_sha256"],
        "sid_n_items": audit["n_items"],
        "sid_item_id_coverage_gaps": audit["coverage_gaps"],
        **clean(l1),
        **{f"l1l2_{k}": v for k, v in clean(l1l2).items() if k != "level"},
        **{f"struct_{k}": v for k, v in struct.items()},
    }
    if model["model"].startswith("TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample") and model[
        "model"
    ].endswith("/resample"):
        row["canonical_baseline_check"] = canonical_check(l1, l1l2)
    return row


def cluster_sufficient_stats(
    pos_match: np.ndarray, neg_match: np.ndarray, anchors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor-cluster sufficient statistics.

    Resampling whole clusters then only needs these three per-cluster sums, which is
    exactly equivalent to materialising every replicate's row indices and is what makes
    2,000 replicates over 16,929 clusters tractable.
    """
    order = np.argsort(anchors, kind="stable")
    sorted_anchors = anchors[order]
    boundaries = np.flatnonzero(np.diff(sorted_anchors)) + 1
    groups = np.split(order, boundaries)
    # pos_match is one boolean per row; neg_match is one boolean per (row, negative), so
    # it must be reduced to a per-row mean before clustering. Resampling clusters of raw
    # negative counts instead of per-row rates silently changes neg_share by a factor of
    # the number of negatives per row.
    neg_row_mean = neg_match.mean(axis=1)
    pos_hits = np.array([pos_match[g].sum() for g in groups], dtype=np.float64)
    neg_hits = np.array([neg_row_mean[g].sum() for g in groups], dtype=np.float64)
    n_rows = np.array([len(g) for g in groups], dtype=np.float64)
    return pos_hits, neg_hits, n_rows


def cluster_bootstrap_from_stats(
    pos_hits: np.ndarray,
    neg_hits: np.ndarray,
    n_rows: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster bootstrap of gap and ER from per-cluster sufficient statistics.

    The RNG is re-seeded per model with the same seed, so every model sees the identical
    sequence of cluster resamples and the intervals are directly comparable.
    """
    rng = np.random.default_rng(seed)
    n_clusters = pos_hits.size
    gaps = np.empty(replicates)
    ers = np.full(replicates, np.nan)
    for b in range(replicates):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        rows = n_rows[picks].sum()
        p = pos_hits[picks].sum() / rows
        neg = neg_hits[picks].sum() / rows
        gaps[b] = p - neg
        if neg > 0:
            ers[b] = p / neg
    finite = ers[np.isfinite(ers)]
    return {
        "n_clusters": int(n_clusters),
        "replicates_defined": int(finite.size),
        "gap_ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
        "er_ci95": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size >= replicates * 0.5
            else ["undefined", "undefined"]
        ),
        "_boot_ers": ers,
        "_boot_gaps": gaps,
    }
