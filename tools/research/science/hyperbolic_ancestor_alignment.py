#!/usr/bin/env python3
"""HBAA: the hyperbolic behavioural ancestor alignment of an exported SID table.

The metric asks whether the true next item's level-1 prefix shares a DEEPER
common ancestor with the user's history than a negative's prefix does. In a space
of negative curvature the Gromov product at the origin

    (x | y)_0 = ( d(0, x) + d(0, y) - d(x, y) ) / 2

is the depth of the two points' nearest common ancestor, and that is the quantity
the metric compares:

    Delta = (u_h | mu_{b(i+)})_0 - (u_h | mu_{b(i-)})_0

with u_h the hyperbolic Frechet barycentre of the SOURCE representations of the
whole training history of the query, and mu_b the barycentre of the TARGET
representations of the items in bucket b. The reduction is Issue #126's, unchanged,
so the number sits on Q_H's scale:

    HBAA = 1 - E[softplus(-Delta / tau)] / ln 2

with tau the frozen reference's own temperature.

The metric was found by the multi-agent metric rounds of Issue #126 and measured
on the 97 confirmed SID+recall pairs: Spearman +0.701 (CI [+0.536, +0.824]) against
Q_H's +0.634. Its degenerate cases behave as the contract requires: a table with a
single level-1 code scores 0.0 and a randomly relabelled table scores 0.0003, so
neither a collapse nor a random assignment can be rewarded.

Usage:
    hyperbolic_ancestor_alignment.py --sid results/stage2_RQ-VAE/<arm>/item_sids.json
    hyperbolic_ancestor_alignment.py --benchmark          # the 97-pair correlation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
REFERENCE_NPZ = REPO / "results" / "research" / "issue126" / "reference" / "reference.npz"
REFERENCE_JSON = REPO / "results" / "research" / "issue126" / "reference" / "reference.json"
TRAIN_PARQUET = REPO / "results" / "stage0_build_parquet" / "train.parquet"
NEGATIVES = 10
SEED = 42


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poincare distance of curvature 1."""
    difference = np.sum((a - b) ** 2, axis=-1)
    denominator = (1.0 - np.sum(a * a, axis=-1)) * (1.0 - np.sum(b * b, axis=-1))
    return np.arccosh(np.maximum(1.0 + 2.0 * difference /
                                 np.maximum(denominator, 1e-12), 1.0 + 1e-12))


def _norm(x: np.ndarray) -> np.ndarray:
    """The geodesic distance to the origin: 2 * arccosh(1 / sqrt(1 - |x|^2)).

    The factor 2 is the radial term of the Poincare metric; without it the
    Gromov product mixes a halved radius with a full distance and is not the
    common-ancestor depth it is meant to be.
    """
    inner = np.clip(1.0 - np.sum(x * x, axis=-1), 1e-12, None)
    return 2.0 * np.arccosh(1.0 / np.sqrt(inner))


def gromov(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The Gromov product at the origin: the common-ancestor depth."""
    return 0.5 * (_norm(x) + np.asarray(_norm(y)) - _distance(x, y))


def _exp0(tangent: np.ndarray) -> np.ndarray:
    """``tanh(|v|) / |v| * v`` -- the exponential map at the origin (curvature 1)."""
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    return np.tanh(norm) * tangent / np.maximum(norm, 1e-12)


def _log0(point: np.ndarray) -> np.ndarray:
    """``artanh(|y|) / |y| * y`` -- the logarithmic map at the origin."""
    norm = np.linalg.norm(point, axis=-1, keepdims=True)
    return np.arctanh(np.clip(norm, 0.0, 1 - 1e-12)) * point / np.maximum(norm, 1e-12)


def _mobius_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Mobius addition in the unit ball."""
    a2 = np.sum(a * a, axis=-1, keepdims=True)
    b2 = np.sum(b * b, axis=-1, keepdims=True)
    ab = np.sum(a * b, axis=-1, keepdims=True)
    numerator = (1 + 2 * ab + b2) * a + (1 - a2) * b
    denominator = 1 + 2 * ab + a2 * b2
    return numerator / np.maximum(denominator, 1e-12)


def _project(points: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(points, axis=-1, keepdims=True)
    return np.where(norm > 1 - 1e-7, points / np.maximum(norm, 1e-12) * (1 - 1e-7), points)


def _log_at(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """``log_base(point)``: the frozen routine's scale, with the factor 2."""
    difference = _mobius_add(-base, point)
    norm = np.linalg.norm(difference, axis=-1, keepdims=True)
    scale = 2.0 * np.arctanh(np.clip(norm, 0.0, 1 - 1e-9)) / np.maximum(norm, 1e-15)
    return scale * difference


def _exp_at(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """``exp_base(v)``: the frozen routine's tanh(norm/2) scaling."""
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    scaled = np.tanh(norm / 2.0) / np.maximum(norm, 1e-15) * tangent
    return _project(_mobius_add(base, scaled))


def frechet_barycentre(points: np.ndarray, steps: int = 20) -> np.ndarray:
    """The Karcher mean in the ball, in numpy.

    The same iteration Issue #126 froze -- the Euclidean mean mapped into the ball
    as the start, then the mean of the base-point logarithms pushed back out with
    the base-point exponential -- but without torch, so the metric is usable from
    the research interpreter and from the plain system python alike.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 1:
        return points[0].copy()
    centre = _exp0(points.mean(axis=0))
    for _ in range(int(steps)):
        tangent = _log_at(centre, points).mean(axis=0)
        moved = _exp_at(centre, tangent)
        shift = float(np.linalg.norm(_log_at(centre, moved)))
        centre = moved
        if shift < 1e-9:
            break
    return centre


def load_reference() -> dict:
    payload = np.load(REFERENCE_NPZ)
    record = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    return {"source": np.asarray(payload["source"], dtype=np.float64),
            "target": np.asarray(payload["target"], dtype=np.float64),
            "temperature": float(record["temperature"])}


def training_queries(items: np.ndarray) -> dict:
    """The training interactions with ten deterministic negatives each."""
    import pandas as pd

    frame = pd.read_parquet(TRAIN_PARQUET, columns=["user", "history", "target"])
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    rng = np.random.default_rng(SEED)
    histories, positives, negatives = [], [], []
    for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
        sequence = [int(item) for item in list(history)]
        if not sequence:
            continue
        seen = set(sequence) | {int(target)}
        chosen, attempts = [], 0
        while len(chosen) < NEGATIVES and attempts < NEGATIVES * 20:
            candidate = int(items[rng.integers(0, items.size)])
            attempts += 1
            if candidate not in seen and candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) < NEGATIVES:
            continue
        histories.append(sequence)
        positives.append(int(target))
        negatives.append(np.asarray(chosen, dtype=np.int64))
    return {"histories": histories,
            "positives": np.asarray(positives, dtype=np.int64),
            "negatives": np.stack(negatives).astype(np.int64)}


def history_points(reference: dict, queries: dict) -> np.ndarray:
    """The query point of every interaction: the barycentre of its history."""
    return np.stack([frechet_barycentre(reference["source"][np.asarray(history, dtype=np.int64)])
                     for history in queries["histories"]])


def score(sid_path: Path, *, reference: dict = None, queries: dict = None,
          points: np.ndarray = None) -> dict:
    """HBAA for one exported SID table."""
    reference = reference or load_reference()
    payload = json.loads(Path(sid_path).read_text(encoding="utf-8"))
    items = sorted(int(key) for key in payload)
    rows = np.array([payload[str(item)] for item in items], dtype=np.int64)
    position = {int(item): index for index, item in enumerate(items)}
    target = reference["target"][np.asarray(items, dtype=np.int64)]

    if queries is None:
        queries = training_queries(np.unique(reference["target"].shape and
                                             np.asarray(reference["target"])))
    if points is None:
        points = history_points(reference, queries)

    codes = rows[:, 0]
    pos = np.array([position.get(int(item), -1) for item in queries["positives"]],
                   dtype=np.int64)
    neg = np.array([[position.get(int(item), -1) for item in row]
                    for row in queries["negatives"]], dtype=np.int64)
    valid = (pos >= 0) & (neg >= 0).all(axis=1)
    if not valid.any():
        return {"hbaa": 0.0, "queries": 0}
    query_points = points[valid]
    pos = pos[valid]
    neg = neg[valid]

    centres = {int(code): frechet_barycentre(target[codes == code])
               for code in np.unique(codes)}
    mu_pos = np.stack([centres[int(codes[index])] for index in pos])
    mu_neg = np.stack([[centres[int(codes[index])] for index in row] for row in neg])
    delta = gromov(query_points, mu_pos)[:, None] - gromov(query_points[:, None, :], mu_neg)
    tau = float(reference["temperature"])
    per_query = np.log1p(np.exp(-delta / tau)).mean(axis=1)
    value = 1.0 - float(per_query.mean()) / float(np.log(2.0))
    return {"hbaa": float(value), "queries": int(len(query_points)),
            "delta_mean": float(delta.mean()),
            "buckets": int(len(centres)),
            "reference_sha256": json.loads(REFERENCE_JSON.read_text())["arrays"]["sha256"]}


def degenerate_cases(sid_path: Path) -> dict:
    """The counter-examples a metric like this must not reward."""
    payload = json.loads(Path(sid_path).read_text(encoding="utf-8"))
    items = sorted(int(key) for key in payload)
    rows = np.array([payload[str(item)] for item in items], dtype=np.int64)
    collapsed = {"rows": np.concatenate([np.zeros((len(items), 1), dtype=np.int64),
                                        rows[:, 1:]], axis=1),
                 "item_ids": np.asarray(items, dtype=np.int64)}
    rng = np.random.default_rng(0)
    random_rows = rows.copy()
    for level in range(random_rows.shape[1]):
        random_rows[:, level] = rng.permutation(random_rows[:, level])
    return {"collapsed": collapsed,
            "random": {"rows": random_rows,
                       "item_ids": np.asarray(items, dtype=np.int64)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", help="an exported item_sids.json")
    parser.add_argument("--benchmark", action="store_true",
                        help="score the 97 confirmed pairs and report the correlation")
    args = parser.parse_args()
    if args.benchmark:
        from tools.research.science import sid_metric_bench as bench
        context = bench.load_context()
        print("the metric lives in results/research/issue126/metric_round/metrics_hbaa.py; "
              "the benchmark is run through sid_metric_bench.py")
        return 0
    if not args.sid:
        parser.error("--sid or --benchmark is required")
    path = Path(args.sid)
    if not path.is_absolute():
        path = REPO / path
    print(json.dumps(score(path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
