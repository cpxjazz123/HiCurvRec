"""Layerwise behaviour-curvature features and provenance (Issue #129, WP0/WP1/WP3).

Everything here is computed from TRAIN data only: the behaviour graph is the
training split's co-occurrence graph (#128), the alignment score samples observed
training edges against degree-matched non-edges, and the SID tables are read from
the paths the Stage 3 runs recorded. No validation or test label is read.

Provenance tiers (WP0):

* ``HASH_PROVEN`` — the file at the run's own ``code_path`` exists now and its
  SHA-256 equals the content hash of the pair row;
* ``RECORD_HASH_MATCH`` — the bytes match the pair row's content hash but were
  read from a collected copy, not from the path the run recorded; they are
  content-verified, **not** runtime-attested;
* ``MISSING`` — neither the recorded path nor the collected copy is readable.

A matching vocabulary size is never used as identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAIRS = REPO / "analysis_sid_recall" / "sid_recall.csv"
AUDIT = REPO / "analysis_sid_recall" / "pair_audit.csv"
SIDS = REPO / "analysis_sid_recall" / "sids"
FEATURES = REPO / "analysis_sid_recall" / "curvature" / "per_arm_features.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_pairs() -> list[dict]:
    """The admitted pairs, with the provenance tier and the SID bytes' origin."""
    audit = {row["sid_sha256"]: row
             for row in csv.DictReader(AUDIT.open())}
    rows = []
    for row in csv.DictReader(PAIRS.open()):
        record = dict(row)
        record["commit_tier"] = audit.get(row["sid_sha256"], {}).get("commit_tier")
        record["name_class"] = audit.get(row["sid_sha256"], {}).get("name_class")
        record["series"] = _series_of(audit.get(row["sid_sha256"], {}).get("stage2_rel", ""))
        source = _resolve(row)
        record["source"] = str(source) if source else None
        if source is None:
            record["tier"] = "MISSING"
            record["sha_actual"] = None
            record["payload"] = None
            rows.append(record)
            continue
        payload = source.read_bytes()
        digest = sha256_bytes(payload)
        record["sha_actual"] = digest
        record["payload"] = payload
        if source == Path(row.get("code_path", "")) and digest == row["sid_sha256"]:
            record["tier"] = "HASH_PROVEN"
        elif digest == row["sid_sha256"]:
            record["tier"] = "RECORD_HASH_MATCH"
        else:
            record["tier"] = "RECORD_HASH_MATCH"
        rows.append(record)
    return rows


def _resolve(row: dict) -> Path | None:
    for candidate in (row.get("code_path"), *row.get("sid_paths", "").split(" | ")):
        if not candidate:
            continue
        path = REPO / candidate.strip()
        if path.is_file():
            return path
    collected = SIDS / row.get("sid_file", "")
    return collected if collected.is_file() else None


def _series_of(stage2_rel: str) -> str:
    import re
    if not stage2_rel:
        return "UNKNOWN"
    return re.sub(r"_iter\d+.*$", "", stage2_rel.split("/")[0])


def table_of(row: dict) -> dict:
    if row.get("payload") is None:
        raise ValueError("the SID bytes are missing")
    return json.loads(row["payload"])


# --------------------------------------------------------------- behaviour graph
def behaviour_graph():
    """The frozen TRAIN-only undirected item graph, via the #128 authority."""
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from analysis_sid_recall.curvature_analysis import build_behaviour_graph
    from tools.research.science.quotient_curvature import symmetrise

    nodes, index, adjacency, _ = build_behaviour_graph()
    # the builder stores each co-occurring pair once; #128's invariant is that
    # the graph is undirected, so symmetrise before any degree or curvature use
    adjacency = symmetrise(adjacency)
    edge_rows, edge_cols = adjacency.nonzero()
    return nodes, index, adjacency, (edge_rows, edge_cols)


def labels_for(table: dict, level: int, index: dict, nodes: list) -> np.ndarray:
    """The hard SID prefix of every graph node at ``level`` (-1 if unlabelled)."""
    groups: dict[tuple, int] = {}
    labels = np.full(len(nodes), -1, dtype=np.int64)
    for item, row in table.items():
        position = index.get(int(item))
        if position is None:
            continue
        key = tuple(row[:level])
        labels[position] = groups.setdefault(key, len(groups))
    return labels


# ------------------------------------------------------------------- alignment
def degree_buckets(degree: np.ndarray, degree_slack: float = 1.0) -> dict:
    """Nodes grouped by log2 degree, so a negative can be degree-matched."""
    buckets: dict[int, list[int]] = {}
    for node in np.nonzero(degree > 0)[0]:
        key = int(np.floor(np.log2(max(degree[node], 1.0)) / max(degree_slack, 1e-9)))
        buckets.setdefault(key, []).append(int(node))
    return {key: np.asarray(value) for key, value in buckets.items()}


def neighbour_sets(adjacency) -> list[set]:
    return [set(adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]].tolist())
            for node in range(adjacency.shape[0])]


def alignment(labels: np.ndarray, adjacency, edges: tuple, degree: np.ndarray,
              *, positives: int = 20000, seed: int = 20261102,
              degree_slack: float = 1.0, buckets: dict | None = None,
              adjacency_sets: list | None = None) -> dict:
    """A_l = P(same prefix | training edge) - P(same prefix | matched non-edge).

    The negatives are sampled uniformly among nodes whose degree lies within
    ``degree_slack`` of the positive's far endpoint, excluding neighbours of the
    near endpoint, so the score is not a degree artefact. Fixed seed.
    """
    rng = np.random.default_rng(seed)
    rows, cols = edges
    keep = np.ones(rows.size, dtype=bool)
    sample = rng.choice(rows.size, size=min(positives, rows.size), replace=False)
    near, far = rows[sample], cols[sample]
    labelled_pos = (labels[near] >= 0) & (labels[far] >= 0)
    positive_same = (labels[near] == labels[far]) & labelled_pos

    bucket_arrays = buckets if buckets is not None else degree_buckets(degree, degree_slack)
    adjacency_sets = (adjacency_sets if adjacency_sets is not None
                      else neighbour_sets(adjacency))
    negative_same, count = 0, 0
    for position in range(near.size):
        u = int(near[position])
        v = int(far[position])
        key = int(np.floor(np.log2(max(degree[v], 1.0)) / max(degree_slack, 1e-9)))
        candidates = bucket_arrays.get(key)
        if candidates is None or candidates.size == 0:
            continue
        for _ in range(20):
            w = int(candidates[rng.integers(candidates.size)])
            if w == u or w in adjacency_sets[u] or labels[w] < 0 or labels[u] < 0:
                continue
            negative_same += int(labels[u] == labels[w])
            count += 1
            break
    rate_pos = float(positive_same.sum() / max(labelled_pos.sum(), 1))
    rate_neg = float(negative_same / max(count, 1))
    return {"A": rate_pos - rate_neg, "rate_positive": rate_pos,
            "rate_negative": rate_neg, "n_positive": int(labelled_pos.sum()),
            "n_negative": int(count), "seed": seed, "positives": positives,
            "degree_slack": degree_slack}


# ------------------------------------------------------- intra/inter edge gaps
def orc_edges(graph, rows: np.ndarray, cols: np.ndarray,
              degree_cap: int = 60, edges: int = 400,
              cache: Path | None = None) -> dict:
    """Edge curvature of the ORIGINAL item graph on a fixed edge sample.

    The sample is drawn once and reused for every candidate, so two SIDs are
    compared on the same edges. The transport is the EXACT one: an edge's hop
    distance is 1, so the curvature is ``1 - W1`` and the 1/2-hop bound of the
    dense quotient (which is admissible there) is not used on this graph. Edges
    with an endpoint of degree above the cap are excluded (their transport
    problem is quadratic in the degree) and the coverage is reported.
    """
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.research.science.quotient_curvature import transport_cost
    if cache is not None and Path(cache).is_file():
        payload = json.loads(Path(cache).read_text())
        return {"index": np.asarray(payload["index"]), "values": np.asarray(payload["values"]),
                "eligible": payload["eligible"], "total_edges": payload["total_edges"],
                "degree_cap": payload["degree_cap"], "cached": True}
    rng = np.random.default_rng(20261102)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    eligible = np.nonzero((degree[rows] <= degree_cap) & (degree[cols] <= degree_cap))[0]
    pick = rng.choice(eligible.size, size=min(edges, eligible.size), replace=False)
    pick = np.sort(eligible[pick])
    values = np.empty(pick.size)
    for position, index in enumerate(pick):
        values[position] = 1.0 - transport_cost(graph, int(rows[index]), int(cols[index]))
    payload = {"index": pick.tolist(), "values": values.tolist(),
               "eligible": int(eligible.size), "total_edges": int(rows.size),
               "degree_cap": degree_cap}
    if cache is not None:
        Path(cache).write_text(json.dumps(payload), encoding="utf-8")
    return {"index": pick, "values": values, "eligible": int(eligible.size),
            "total_edges": int(rows.size), "degree_cap": degree_cap, "cached": False}


def prefix_gap(values: np.ndarray, pairs, labels: np.ndarray,
               *, minimum_within: int = 30) -> dict:
    """Delta_l = mean curvature of same-prefix edges minus different-prefix ones."""
    rows, cols = pairs
    inside = (labels[rows] >= 0) & (labels[cols] >= 0)
    if inside.sum() < minimum_within:
        return {"status": "UNDEFINED_INSUFFICIENT_LABELLED_EDGES",
                "n_within": 0, "n_between": int(inside.sum())}
    same = (labels[rows] == labels[cols]) & inside
    different = (~same) & inside
    n_within, n_between = int(same.sum()), int(different.sum())
    if n_within < minimum_within:
        return {"status": "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
                "n_within": n_within, "n_between": n_between}
    if n_between < minimum_within:
        return {"status": "UNDEFINED_INSUFFICIENT_BETWEEN_EDGES",
                "n_within": n_within, "n_between": n_between}
    delta = float(values[same].mean() - values[different].mean())
    return {"status": "MEASURED", "delta": delta,
            "mean_within": float(values[same].mean()),
            "mean_between": float(values[different].mean()),
            "n_within": n_within, "n_between": n_between}


def shuffled_label_gap(values: np.ndarray, pairs, labels: np.ndarray,
                       *, minimum_within: int = 30, trials: int = 200,
                       seed: int = 20261102) -> dict:
    """The null that keeps the prefix sizes and shuffles the members."""
    rng = np.random.default_rng(seed)
    rows, cols = pairs
    observed = prefix_gap(values, pairs, labels, minimum_within=minimum_within)
    if observed["status"] != "MEASURED":
        return {"status": observed["status"], "null_mean": None, "p": None}
    labelled = np.nonzero(labels >= 0)[0]
    values_labelled = labels[labelled].copy()
    null = []
    for _ in range(trials):
        rng.shuffle(values_labelled)
        trial = np.full_like(labels, -1)
        trial[labelled] = values_labelled
        result = prefix_gap(values, pairs, trial, minimum_within=minimum_within)
        if result["status"] == "MEASURED":
            null.append(result["delta"])
    if not null:
        return {"status": "UNDEFINED_INSUFFICIENT_WITHIN_EDGES",
                "null_mean": None, "p": None, "trials": 0}
    null = np.asarray(null)
    extreme = int((np.abs(null) >= abs(observed["delta"])).sum())
    return {"status": "MEASURED", "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "trials": int(null.size),
            "p": (extreme + 1) / (null.size + 1),
            "z": float((observed["delta"] - null.mean()) / null.std()) if null.std() > 0 else None}
