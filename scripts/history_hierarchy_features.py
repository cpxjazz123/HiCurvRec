"""Frozen-model feature layer for issue #48.

Everything here is read-only with respect to the 38 frozen arms: it reads
``model_inventory.json``, the #47 manifest, the held-out valid pairs and each arm's
verified ``item_sids.json``, and returns new Stage 2 proxies.

Two of the three preregistered features live here:

  F2  CHRD -- conditional hierarchical refinement discrimination.
      ``D2 = P(same L1+L2 | pair already shares L1, positive) -
             P(same L1+L2 | pair already shares L1, negative)``.
      This is a conditional difference, NOT ``ER_L1L2`` and NOT ``Gap_L1``. The
      conditioning sets and their sizes are reported so a thin conditioning set is
      visible; when a set is empty the feature is NA with a reason, never imputed.

  F3  TWA -- target-weighted SID ambiguity.
      ``m(y) = #{i in catalogue : SID_{1:3}(i) == SID_{1:3}(y)}`` and
      ``A_target = mean over the held-out valid TARGETS of log m(y)``.
      Targets are weighted by how often they are asked for, which is not the same
      as averaging over catalogue items; the catalogue-uniform version is reported
      alongside for contrast.

The fourth token is the collision-extension token. It takes no part in any
three-token ambiguity, prefix match, or collision figure here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"
PRIOR_MANIFEST = (
    REPO_ROOT / "results/stage3_T5Train/monotonicity/joint_prefix_predictors/manifest.json"
)
PRIOR_FEATURES = (
    REPO_ROOT / "results/stage3_T5Train/monotonicity/joint_prefix_predictors/stage2_valid_features.csv"
)
VALID_PAIRS = REPO_ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
SID_STAGING = Path("/tmp/r3_recover")

TRAIN_PARQUET = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
VALID_PARQUET = REPO_ROOT / "results/stage0_build_parquet/valid.parquet"
TEST_PARQUET = REPO_ROOT / "results/stage0_build_parquet/test.parquet"
ITEMS_PARQUET = REPO_ROOT / "results/stage0_build_parquet/items.parquet"

BASELINE_L1_ER = 16.38953942232631
BASELINE_L1L2_ER = 58.38709677419355

HISTORY_TRUNCATION = 20
CODEBOOK_CAPACITY = 256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_sid_path(sid_path: str) -> Path | None:
    direct = REPO_ROOT / sid_path
    if direct.is_file():
        return direct
    staged = SID_STAGING / sid_path
    if staged.is_file():
        return staged
    return None


def load_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY.read_text())


def load_sid(path: Path) -> tuple[np.ndarray, str]:
    """Returns (matrix[n_items, 4], sha256). Row index is the raw item id."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    max_id = max(int(k) for k in payload)
    matrix = np.full((max_id + 1, 4), -1, dtype=np.int64)
    for key, row in payload.items():
        matrix[int(key)] = row
    return matrix, sha


def load_valid_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(VALID_PAIRS)
    return (
        d["anchors"].astype(np.int64),
        d["positives"].astype(np.int64),
        d["negatives"].astype(np.int64),
        d["anchors"].astype(np.int64),
    )


# --------------------------------------------------------------------------- F2
def chrd_features(
    sid: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
) -> dict[str, Any]:
    """Conditional hierarchical refinement discrimination, L2 primary and L3 exploratory."""
    out: dict[str, Any] = {}
    for level in (2, 3):
        shared1_a = sid[anchors, 0]
        same1_pos = sid[positives, 0] == shared1_a                      # [N]
        same1_neg = sid[negatives, 0] == shared1_a[:, None]            # [N,10]

        # Condition on "already shares L1", then ask about L2 (or L1+L2 .. L3).
        if level == 2:
            cond_pos = same1_pos
            same_prefix_pos = (sid[positives, 1] == sid[anchors, 1])
            cond_neg = same1_neg
            same_prefix_neg = (sid[negatives, 1] == sid[anchors, 1][:, None])
        else:
            shared2_a = sid[anchors, 1]
            cond_pos = same1_pos & (sid[positives, 1] == shared2_a)
            same_prefix_pos = sid[positives, 2] == sid[anchors, 2]
            cond_neg = same1_neg & (sid[negatives, 1] == shared2_a[:, None])
            same_prefix_neg = sid[negatives, 2] == sid[anchors, 2][:, None]

        pos_n = int(cond_pos.sum())
        neg_n = int(cond_neg.sum())
        pos_hits = int((cond_pos & same_prefix_pos).sum())
        neg_hits = int((cond_neg & same_prefix_neg).sum())

        r2_pos = pos_hits / pos_n if pos_n else None
        r2_neg = neg_hits / neg_n if neg_n else None

        if r2_pos is None or r2_neg is None:
            d2: float | str = "NA"
            rr: float | str = "NA"
            reason = "empty_conditioning_set"
        else:
            d2 = float(r2_pos - r2_neg)
            # The conditional rate ratio is only reported with a non-zero, adequately
            # supported denominator; otherwise it is NA rather than a huge number.
            if neg_hits >= 30:
                rr = float(r2_pos / r2_neg) if r2_neg > 0 else "NA_zero_denominator"
            else:
                rr = "NA_thin_denominator"
            reason = "ok"

        out[f"chrd_D{level}"] = d2
        out[f"chrd_D{level}_status"] = reason
        out[f"chrd_r{level}_cond_pos"] = r2_pos
        out[f"chrd_r{level}_cond_neg"] = r2_neg
        out[f"chrd_r{level}_ratio"] = rr
        out[f"chrd_cond_pos_n{level}"] = pos_n
        out[f"chrd_cond_neg_n{level}"] = neg_n
        out[f"chrd_cond_pos_hits{level}"] = pos_hits
        out[f"chrd_cond_neg_hits{level}"] = neg_hits
    return out


def chrd_cluster_bootstrap(
    sid: np.ndarray,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    cluster_of_row: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Anchor-cluster paired bootstrap of D2: both arms' terms resample the same rows."""
    shared1_a = sid[anchors, 0]
    cond_pos = sid[positives, 0] == shared1_a
    hit_pos = cond_pos & (sid[positives, 1] == sid[anchors, 1])
    cond_neg = sid[negatives, 0] == shared1_a[:, None]
    hit_neg = cond_neg & (sid[negatives, 1] == sid[anchors, 1][:, None])

    per_row_pos_n = cond_pos.astype(np.float64)
    per_row_pos_hits = hit_pos.astype(np.float64)
    per_row_neg_n = cond_neg.sum(axis=1).astype(np.float64)
    per_row_neg_hits = hit_neg.sum(axis=1).astype(np.float64)

    order = np.argsort(cluster_of_row, kind="stable")
    sorted_clusters = cluster_of_row[order]
    bounds = np.flatnonzero(np.diff(sorted_clusters)) + 1
    groups = np.split(order, bounds)

    def d2_from(idx: np.ndarray) -> float:
        p_hits = per_row_pos_hits[idx].sum()
        p_n = per_row_pos_n[idx].sum()
        n_hits = per_row_neg_hits[idx].sum()
        n_n = per_row_neg_n[idx].sum()
        if p_n <= 0 or n_n <= 0:
            return np.nan
        return float((p_hits / p_n) - (n_hits / n_n))

    point = d2_from(np.arange(len(anchors)))
    rng = np.random.default_rng(seed)
    n_clusters = len(groups)
    draws = np.empty(replicates)
    for b in range(replicates):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        draws[b] = d2_from(np.concatenate([groups[c] for c in picks]))
    finite = draws[np.isfinite(draws)]
    return {
        "point": point,
        "ci95": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size >= replicates * 0.5
            else ["undefined", "undefined"]
        ),
        "replicates_defined": int(finite.size),
        "n_clusters": int(n_clusters),
    }


# --------------------------------------------------------------------------- F3
def target_ambiguity(
    sid: np.ndarray,
    valid_targets: np.ndarray,
    catalogue_ids: np.ndarray,
) -> dict[str, Any]:
    """F3: ambiguity weighted by how often an item is asked for as a valid target."""
    code = np.full(sid.shape[0], -1, dtype=np.int64)
    code[catalogue_ids] = (
        sid[catalogue_ids, 0] * CODEBOOK_CAPACITY * CODEBOOK_CAPACITY
        + sid[catalogue_ids, 1] * CODEBOOK_CAPACITY
        + sid[catalogue_ids, 2]
    )
    group_size = np.bincount(code[catalogue_ids], minlength=CODEBOOK_CAPACITY ** 3)
    m = group_size[code[valid_targets]]
    log_m = np.log(m)
    cat_m = group_size[group_size > 0]

    return {
        "A_target": float(log_m.mean()),
        "A_target_std": float(log_m.std()),
        "amb_p_gt1": float((m > 1).mean()),
        "amb_mean_group_size": float(m.mean()),
        "amb_p90_group_size": float(np.quantile(m, 0.90)),
        "amb_p99_group_size": float(np.quantile(m, 0.99)),
        "amb_max_group_size": int(m.max()),
        "A_catalogue_uniform": float(np.log(cat_m).mean()),
        "amb_distinct_catalogue_codes": int(cat_m.size),
        "amb_valid_target_rows": int(len(valid_targets)),
        "amb_distinct_valid_targets": int(len(np.unique(valid_targets))),
    }


def load_valid_target_ids() -> np.ndarray:
    """Target item ids of the held-out valid pairs, in pair order, with repeats kept.

    F3 weights a target by how often it is actually asked for, so duplicates are
    intentional and are NOT deduplicated.
    """
    import pandas as pd

    df = pd.read_parquet(VALID_PARQUET, columns=["seen_history", "target"])
    rows = [
        (int(h[-1]), int(t))
        for h, t in zip(df["seen_history"].to_numpy(), df["target"].to_numpy(dtype=np.int64))
        if h is not None and len(h) > 0
    ]
    train = pd.read_parquet(TRAIN_PARQUET, columns=["seen_history", "target"])
    train_pairs = {
        (int(h[-1]), int(t))
        for h, t in zip(train["seen_history"].to_numpy(), train["target"].to_numpy(dtype=np.int64))
        if h is not None and len(h) > 0
    }
    held = np.array([t for (_, t) in rows if _ not in train_pairs], dtype=np.int64)
    return held


def load_catalogue_ids() -> np.ndarray:
    items = __import__("pandas").read_parquet(ITEMS_PARQUET, columns=["item_id"])
    return items["item_id"].to_numpy(dtype=np.int64)
